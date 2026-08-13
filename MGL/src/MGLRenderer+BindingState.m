// MGLRenderer+BindingState.m
// Vertex/fragment buffer, attribute and texture binding methods
// extracted from MGLRenderer+Draw.m

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "mgl_frame_activity.h"
#include "mgl_env_flag.h"
#include "mgl_render_cpp.h"

static BOOL mglBindingStateUsesMetalCpp(void)
{
    return mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
           mglRenderCppGetDevice() != NULL;
}

static id<MTLBuffer> mglBindingStateCreateBuffer(
    id<MTLDevice> device,
    NSUInteger length,
    MTLResourceOptions options)
{
    if (mglBindingStateUsesMetalCpp()) {
        void *buffer = NULL;
        if (mglRenderCppCreateBuffer(length, options, NULL, &buffer) == 0 &&
            buffer) {
            return (__bridge_transfer id<MTLBuffer>)buffer;
        }
    }
    return [device newBufferWithLength:length options:options];
}

static id<MTLBuffer> mglBindingStateCreateBufferWithBytes(
    id<MTLDevice> device,
    const void *bytes,
    NSUInteger length,
    MTLResourceOptions options)
{
    if (mglBindingStateUsesMetalCpp()) {
        void *buffer = NULL;
        if (mglRenderCppCreateBufferWithBytes(bytes, length, options, NULL,
                                              &buffer) == 0 && buffer) {
            return (__bridge_transfer id<MTLBuffer>)buffer;
        }
    }
    return [device newBufferWithBytes:bytes length:length options:options];
}

static id<MTLTexture> mglBindingStateCreateTextureLevelView(
    id<MTLTexture> texture,
    NSUInteger level,
    NSUInteger sliceCount)
{
    if (mglBindingStateUsesMetalCpp()) {
        void *view = NULL;
        if (mglRenderCppCreateTextureViewRange(
                (__bridge void *)texture, (uint32_t)texture.pixelFormat,
                (uint32_t)texture.textureType, level, 1, 0, sliceCount,
                0, 0, 0, 0, 0, &view) == 0 && view) {
            return (__bridge_transfer id<MTLTexture>)view;
        }
    }
    return [texture newTextureViewWithPixelFormat:texture.pixelFormat
                                      textureType:texture.textureType
                                           levels:NSMakeRange(level, 1)
                                           slices:NSMakeRange(0, sliceCount)];
}

static void mglBindingStateSetVertexBuffer(
    id<MTLRenderCommandEncoder> encoder,
    id<MTLBuffer> buffer,
    NSUInteger offset,
    NSUInteger index)
{
    if (mglBindingStateUsesMetalCpp() &&
        mglRenderCppSetRenderBuffer(
            (__bridge void *)encoder, (__bridge void *)buffer, offset,
            MGL_RENDER_CPP_BINDING_STAGE_VERTEX, (uint32_t)index) == 0) {
        return;
    }
    [encoder setVertexBuffer:buffer offset:offset atIndex:index];
}

static void mglBindingStateSetVertexBytes(
    id<MTLRenderCommandEncoder> encoder,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    if (mglBindingStateUsesMetalCpp() &&
        mglRenderCppSetRenderBytes((__bridge void *)encoder, bytes, length,
                                   MGL_RENDER_CPP_BINDING_STAGE_VERTEX,
                                   (uint32_t)index) == 0) {
        return;
    }
    [encoder setVertexBytes:bytes length:length atIndex:index];
}

static void mglBindingStateSetFragmentBuffer(
    id<MTLRenderCommandEncoder> encoder,
    id<MTLBuffer> buffer,
    NSUInteger offset,
    NSUInteger index)
{
    if (mglBindingStateUsesMetalCpp() &&
        mglRenderCppSetRenderBuffer(
            (__bridge void *)encoder, (__bridge void *)buffer, offset,
            MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT, (uint32_t)index) == 0) {
        return;
    }
    [encoder setFragmentBuffer:buffer offset:offset atIndex:index];
}

static void mglBindingStateSetFragmentBytes(
    id<MTLRenderCommandEncoder> encoder,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    if (mglBindingStateUsesMetalCpp() &&
        mglRenderCppSetRenderBytes((__bridge void *)encoder, bytes, length,
                                   MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT,
                                   (uint32_t)index) == 0) {
        return;
    }
    [encoder setFragmentBytes:bytes length:length atIndex:index];
}

@implementation MGLRenderer (Draw)

- (bool) bindVertexBuffersToCurrentRenderEncoder:(const MGLEncodeContext *)encCtx
{
    static uint64_t s_vbindCallCount = 0;
    static double s_vbindLastCallTime = 0.0;
    static uint64_t s_vbindLastCallCount = 0;
    uint64_t vbindCall = ++s_vbindCallCount;
    double vbindStartSeconds = mglTraceNowSeconds();
    uint64_t vbindStartNS = mglTraceClockNS();
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
              ctx, ctx ? MGL_STATE(ctx)->vao : NULL, encCtx->encoder);
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
    activeProgram = _tessellation.nativeTESActive
        ? _tessellation.nativeTESProgram
        : mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    const int vertexStage = _tessellation.nativeTESActive
        ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;

    if (kMGLVerboseBindLogs) {
        NSLog(@"MGL VBIND vao=%p magic=0x%x", vao, vao->magic);
    }
    mapCount = MGL_STATE(ctx)->vertex_buffer_map_list.count;
    if (mapCount > MAX_MAPPED_BUFFERS) {
        static uint64_t s_vbindMapCountOverflow = 0;
        uint64_t hit = ++s_vbindMapCountOverflow;
        if (hit <= 16ull || (hit % 4096ull) == 0ull) {
            NSLog(@"MGL WARNING: VBIND mapCount=%u exceeds MAX_MAPPED_BUFFERS=%d, clamping (hit=%llu)",
                  mapCount, MAX_MAPPED_BUFFERS, (unsigned long long)hit);
        }
        mapCount = MAX_MAPPED_BUFFERS;
    }

    for (GLuint i = 0; i < MAX_ATTRIBS; i++) {
        attribBindingIndex[i] = -1;
    }

    // Resolve attribute slot reservations first so base/resource bindings do not
    // overwrite shader-required vertex input slots.
    bool attribsEnabledByApp = (vao->enabled_attribs != 0u);
    GLuint reserveMaxAttribs = MAX_ATTRIBS;
    for (GLuint attrib = 0; attrib < reserveMaxAttribs; attrib++) {
        if (!mglRendererProgramUsesVertexAttrib(activeProgram, attrib)) {
            continue;
        }

        int mappedIndex = [self getVertexBufferIndexWithAttributeSet:(int)attrib];
        if (mappedIndex < 0 || mappedIndex >= (int)kMGLMaxMetalVertexBufferCount) {
            NSLog(@"MGL ERROR: VBIND reserve attrib=%u unresolved mapping=%d", attrib, mappedIndex);
            continue;
        }

        attribBindingIndex[attrib] = mappedIndex;
        attribBindingReserved[mappedIndex] = true;
    }

    if (kMGLVerboseBindLogs) {
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

            if (enabled && attribBuffer) {
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
    }

    for(int i=0; i<(int)mapCount; i++)
    {
        map = &MGL_STATE(ctx)->vertex_buffer_map_list.buffers[i];
        
        ptr = mglRendererGetValidatedBuffer(ctx, map->buf, __FUNCTION__, (NSUInteger)i);
        offset = map->offset;
        isBaseBinding = (map->attribute_mask == 0);
        GLuint glBindingIndex = map->buffer_base_index;
        bindingIndex = glBindingIndex;
        if (isBaseBinding) {
            NSInteger metalBindingIndex = map->has_metal_binding
                ? (NSInteger)map->metal_binding_index
                : [self getProgramMetalBufferIndexForStage:vertexStage
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
            mglBindingStateSetVertexBuffer(encCtx->encoder, nil, 0, bindingIndex);
            mglRenderCppBindingClearVertexBuffer(_bindingStateOwner,
                                                  (uint32_t)bindingIndex);
            continue;
        }

        if (offset < 0) {
            NSLog(@"MGL WARNING: Vertex buffer map[%d] has negative offset=%lld, skipping",
                  i, (long long)offset);
            mglBindingStateSetVertexBuffer(encCtx->encoder, nil, 0, bindingIndex);
            mglRenderCppBindingClearVertexBuffer(_bindingStateOwner,
                                                  (uint32_t)bindingIndex);
            continue;
        }

        if (ptr->size < 0) {
            NSLog(@"MGL WARNING: Vertex buffer %u has invalid size=%lld, skipping",
                  ptr->name, (long long)ptr->size);
            mglBindingStateSetVertexBuffer(encCtx->encoder, nil, 0, bindingIndex);
            mglRenderCppBindingClearVertexBuffer(_bindingStateOwner,
                                                  (uint32_t)bindingIndex);
            continue;
        }

        NSUInteger bindOffset = (NSUInteger)offset;
        NSUInteger reflectedRequiredBytes = 0;
        NSUInteger requiredBindingBytes = kMGLMinimumStageBindingSize;
        if (isBaseBinding && glBindingIndex < MAX_BINDABLE_BUFFERS) {
            reflectedRequiredBytes = map->has_metal_binding
                ? [self getProgramBindingRequiredSize:vertexStage
                                                 type:(int)map->resource_type
                                                index:(int)map->resource_index]
                : [self getProgramBindingRequiredSizeForStage:vertexStage
                                                clientBinding:glBindingIndex];
            if (reflectedRequiredBytes > requiredBindingBytes) {
                requiredBindingBytes = reflectedRequiredBytes;
            }
        }

        /* For small uniform constants (plain uniforms), use setVertexBytes
         * to copy the data into the command buffer at bind time. This is
         * critical for correctness when the same uniform buffer is updated
         * between draws encoded into the same command buffer — a shared-
         * memory MTLBuffer would let the GPU see only the final value.
         *
         * Decided before bindMTLBuffer so these slots never materialize an
         * MTLBuffer at all: with one, every glUniform* upload allocates a
         * copy-on-write snapshot, and a slot below requiredBindingBytes then
         * allocates a zero-padded isolated buffer per draw on top of it.
         * Padding here reproduces exactly what that isolated buffer held. */
        if (isBaseBinding &&
            map->resource_type == _UNIFORM_CONSTANT_RES &&
            ptr->data.buffer_data &&
            offset == 0 &&
            requiredBindingBytes <= kMGLStageBindingStackScratchSize) {
            NSUInteger visibleBytes =
                (NSUInteger)mglBufferMapVisibleBackingBytes(map, ptr->data.buffer_size);
            NSUInteger inlineLength = MAX(visibleBytes, requiredBindingBytes);
            if (visibleBytes > 0 && inlineLength <= kMGLStageBindingStackScratchSize) {
                uint8_t padded[kMGLStageBindingStackScratchSize];
                const void *inlineBytes = (const void *)(uintptr_t)ptr->data.buffer_data;
                if (inlineLength > visibleBytes) {
                    memcpy(padded, inlineBytes, visibleBytes);
                    memset(padded + visibleBytes, 0, inlineLength - visibleBytes);
                    inlineBytes = padded;
                }
                mglBindingStateSetVertexBytes(encCtx->encoder, inlineBytes,
                                              inlineLength, bindingIndex);
                [self invalidateLastBoundVertexBufferAtIndex:bindingIndex];
                anyBindingPresent[bindingIndex] = true;
                /* Only clear while no MTLBuffer exists: an existing one would
                 * keep stale contents with no pending upload left to fix it. */
                if (!ptr->data.mtl_data) {
                    ptr->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
                }
                if (kMGLVerboseBindLogs) {
                    NSLog(@"MGL VBIND uniform-constant setVertexBytes slot=%lu buffer=%u len=%lu visible=%lu",
                          (unsigned long)bindingIndex,
                          ptr->name,
                          (unsigned long)inlineLength,
                          (unsigned long)visibleBytes);
                }
                continue;
            }
        }

        if (!ptr->data.mtl_data) {
            [self bindMTLBuffer:ptr];
        } else if (ptr->data.dirty_bits & (DIRTY_BUFFER_DATA | DIRTY_BUFFER_ADDR)) {
            /* A CPU write (map/unmap, BufferSubData) since the Metal backing
             * was created is normally pushed by updateDirtyBaseBufferList.
             * On the first draw the base map list is still empty when that
             * path runs, so refresh here before binding stale contents. */
            [self updateDirtyBuffer:ptr];
        }
        id<MTLBuffer> buffer = nil;
        if (ptr->data.mtl_data &&
            (uintptr_t)ptr->data.mtl_data >= 0x10000u) {
            buffer = (__bridge id<MTLBuffer>)(ptr->data.mtl_data);
        }

        NSUInteger metalLen = buffer ? buffer.length : 0u;
        NSUInteger availableBytes = buffer
            ? mglBufferMapVisibleBackingBytes(map, metalLen)
            : 0u;

        BOOL needsIsolatedBinding =
            !buffer || bindOffset >= metalLen ||
            availableBytes < requiredBindingBytes;
        if (needsIsolatedBinding &&
            (!ptr->gpu_write_target || _tessellation.nativeTESActive)) {
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
                mglBindingStateSetVertexBuffer(encCtx->encoder, nil, 0,
                                               bindingIndex);
                mglRenderCppBindingClearVertexBuffer(_bindingStateOwner,
                                                      (uint32_t)bindingIndex);
                continue;
            }

            BOOL writableResource =
                map->resource_type == _STORAGE_BUFFER_RES ||
                map->resource_type == _ATOMIC_COUNTER_RES;
            if (_tessellation.nativeTESActive && writableResource && buffer &&
                availableBytes > 0 &&
                ![self recordStageBindingCopyBack:
                           &_tessellation.nativeTESCopyBacks
                                             atIndex:bindingIndex
                                           temporary:isolated
                                         destination:buffer
                                   destinationBuffer:ptr
                                  destinationOffset:bindOffset
                                              length:availableBytes]) {
                return false;
            }

            mglBindingStateSetVertexBuffer(encCtx->encoder, isolated, 0,
                                           bindingIndex);
            mglRenderCppBindingUpdateVertexBuffer(
                _bindingStateOwner, (__bridge void *)isolated, 0,
                (uint32_t)bindingIndex);
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

        if (!mglBindingStateIsValid(_bindingStateOwner) ||
            !mglBindingStateBufferMatches(
                _bindingStateOwner, MGL_RENDER_CPP_BINDING_STAGE_VERTEX,
                (__bridge void *)buffer, (NSUInteger)offset,
                (uint32_t)bindingIndex)) {
            mglBindingStateSetVertexBuffer(encCtx->encoder, buffer,
                                           (NSUInteger)offset, bindingIndex);
            mglRenderCppBindingUpdateVertexBuffer(
                _bindingStateOwner, (__bridge void *)buffer,
                (NSUInteger)offset, (uint32_t)bindingIndex);
            MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
            mglNoteBufferEncoded(ptr);
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
    mglRenderCppBindingOrVertexBufferMask(_bindingStateOwner,
                                          boundVertexBufferMask);

    if (mglEnvFlagEnabled("MGL_TRACE_SPARSE_BINDING")) {
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
        mglTraceLogNSString(@"MGL TRACE vbind.end call=%llu mapCount=%u boundSlots=%lu reservedAttribSlots=%lu baseSlots=%lu elapsed=%.1fus",
              (unsigned long long)vbindCall,
              (unsigned)mapCount,
              (unsigned long)boundSlots,
              (unsigned long)reservedSlots,
              (unsigned long)baseSlots,
              (mglTraceClockNS() - vbindStartNS) / 1000.0);
    }

    /* Mark the dedup cache as valid for the current encoder so subsequent
     * binds can be skipped when the resource and offset are unchanged. */
    mglRenderCppBindingSetValid(_bindingStateOwner, 1);
    return true;
}

/* Bind vertex attributes from the VAO.  Extracted from
 * bindVertexBuffersToCurrentRenderEncoder to keep that function under the
 * 500-line limit.  Pure mechanical extraction — no behavior change. */
- (bool)bindVertexAttributesFromVAO:(VertexArray *)vao
                      activeProgram:(Program *)activeProgram
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
            NSUInteger totalByteCount = kMGLCurrentAttribRepeatCount * attribStride;

            /* Reuse the cached MTLBuffer when the current vertex attrib value
             * and stride haven't changed since the last draw.  This avoids the
             * per-draw NSMutableData allocation + newBufferWithBytes + 4096×
             * memcpy loop. */
            id<MTLBuffer> currentAttribBuffer = nil;
            if (_currentAttribCacheValid[attrib] &&
                _currentAttribCacheStride[attrib] == attribStride &&
                memcmp(_currentAttribCacheBytes[attrib], attribBytes, attribStride) == 0 &&
                _currentAttribBuffers[attrib] != nil) {
                currentAttribBuffer = _currentAttribBuffers[attrib];
            }

            if (currentAttribBuffer == nil) {
                /* Cache miss — rebuild the repeated buffer. */
                NSMutableData *repeated = [NSMutableData dataWithLength:totalByteCount];
                if (!repeated) {
                    NSLog(@"MGL VBIND skip attrib=%u: failed to allocate current vertex attrib stream", attrib);
                    continue;
                }
                uint8_t *dst = (uint8_t *)repeated.mutableBytes;
                for (NSUInteger v = 0; v < kMGLCurrentAttribRepeatCount; v++) {
                    memcpy(dst + v * attribStride, attribBytes, MIN((NSUInteger)16u, attribStride));
                }
                currentAttribBuffer = mglBindingStateCreateBufferWithBytes(
                    _device, repeated.bytes, repeated.length,
                    MTLResourceStorageModeShared);
                if (!currentAttribBuffer) {
                    NSLog(@"MGL VBIND skip attrib=%u: failed to allocate current vertex attrib Metal buffer", attrib);
                    continue;
                }
                /* Store in cache. */
                _currentAttribBuffers[attrib] = currentAttribBuffer;
                memcpy(_currentAttribCacheBytes[attrib], attribBytes, attribStride);
                _currentAttribCacheStride[attrib] = attribStride;
                _currentAttribCacheValid[attrib] = YES;
            }
            if (!mglBindingStateIsValid(_bindingStateOwner) ||
                !mglBindingStateBufferMatches(
                    _bindingStateOwner, MGL_RENDER_CPP_BINDING_STAGE_VERTEX,
                    (__bridge void *)currentAttribBuffer, 0, (uint32_t)bindingIndex)) {
                mglBindingStateSetVertexBuffer(encCtx->encoder,
                                               currentAttribBuffer, 0,
                                               bindingIndex);
                mglRenderCppBindingUpdateVertexBuffer(
                    _bindingStateOwner, (__bridge void *)currentAttribBuffer, 0,
                    (uint32_t)bindingIndex);
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
                            (int)MGL_STATE(ctx)->current_vertex_attrib[attrib].i[0],
                            (int)MGL_STATE(ctx)->current_vertex_attrib[attrib].i[1],
                            (int)MGL_STATE(ctx)->current_vertex_attrib[attrib].i[2],
                            (int)MGL_STATE(ctx)->current_vertex_attrib[attrib].i[3],
                            MGL_STATE(ctx)->current_vertex_attrib[attrib].f[0],
                            MGL_STATE(ctx)->current_vertex_attrib[attrib].f[1],
                            MGL_STATE(ctx)->current_vertex_attrib[attrib].f[2],
                            MGL_STATE(ctx)->current_vertex_attrib[attrib].f[3]);
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

        /* GL_FIXED / GL_UNSIGNED_INT_10_10_10_2 / GL_UNSIGNED_INT_10F_11F_11F_REV
         * have no direct Metal vertex format (see glTypeSizeToMtlType). They
         * are unpacked to float on the CPU, mirroring the GL_DOUBLE path. */
        bool needsPackedConversion = (attribState->type == GL_FIXED ||
                                      attribState->type == GL_UNSIGNED_INT_10_10_10_2 ||
                                      attribState->type == GL_UNSIGNED_INT_10F_11F_11F_REV);

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
            !needsIntegerConversion && !needsPackedConversion && anyBindingPresent[bindingIndex]) {
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
            if (!mglBindingStateIsValid(_bindingStateOwner) ||
                !mglBindingStateBufferMatches(
                    _bindingStateOwner, MGL_RENDER_CPP_BINDING_STAGE_VERTEX,
                    (__bridge void *)convertedBuffer, 0, (uint32_t)bindingIndex)) {
                mglBindingStateSetVertexBuffer(encCtx->encoder, convertedBuffer,
                                               0, bindingIndex);
                mglRenderCppBindingUpdateVertexBuffer(
                    _bindingStateOwner, (__bridge void *)convertedBuffer, 0,
                    (uint32_t)bindingIndex);
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
            if (!mglBindingStateIsValid(_bindingStateOwner) ||
                !mglBindingStateBufferMatches(
                    _bindingStateOwner, MGL_RENDER_CPP_BINDING_STAGE_VERTEX,
                    (__bridge void *)convertedBuffer, 0, (uint32_t)bindingIndex)) {
                mglBindingStateSetVertexBuffer(encCtx->encoder, convertedBuffer,
                                               0, bindingIndex);
                mglRenderCppBindingUpdateVertexBuffer(
                    _bindingStateOwner, (__bridge void *)convertedBuffer, 0,
                    (uint32_t)bindingIndex);
                MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
            } else {
                MGL_PERF_INC(g_mglSetVertexBufferSkipsSinceSwap);
            }
            anyBindingPresent[bindingIndex] = true;
            continue;
        }

        if (needsPackedConversion) {
            NSUInteger convertedStride = 0;
            id<MTLBuffer> convertedBuffer = nil;
            if (attribState->type == GL_FIXED) {
                convertedBuffer = [self floatVertexBufferForFixedAttrib:attribBuffer
                                                               resolved:&resolved
                                                                   size:attribState->size
                                                              outStride:&convertedStride];
            } else if (attribState->type == GL_UNSIGNED_INT_10_10_10_2) {
                convertedBuffer = [self floatVertexBufferForPacked1010102Attrib:attribBuffer
                                                                        resolved:&resolved
                                                                       outStride:&convertedStride];
            } else { /* GL_UNSIGNED_INT_10F_11F_11F_REV */
                convertedBuffer = [self floatVertexBufferForPacked10f11f11fAttrib:attribBuffer
                                                                           resolved:&resolved
                                                                          outStride:&convertedStride];
            }
            if (!convertedBuffer) {
                NSLog(@"MGL VBIND skip attrib=%u buffer=%u: failed to convert packed/fixed vertex attrib (type=0x%x)",
                      attrib, attribBuffer->name, (unsigned)attribState->type);
                continue;
            }
            if (!mglBindingStateIsValid(_bindingStateOwner) ||
                !mglBindingStateBufferMatches(
                    _bindingStateOwner, MGL_RENDER_CPP_BINDING_STAGE_VERTEX,
                    (__bridge void *)convertedBuffer, 0, (uint32_t)bindingIndex)) {
                mglBindingStateSetVertexBuffer(encCtx->encoder, convertedBuffer,
                                               0, bindingIndex);
                mglRenderCppBindingUpdateVertexBuffer(
                    _bindingStateOwner, (__bridge void *)convertedBuffer, 0,
                    (uint32_t)bindingIndex);
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
            if (!mglBindingStateIsValid(_bindingStateOwner) ||
                !mglBindingStateBufferMatches(
                    _bindingStateOwner, MGL_RENDER_CPP_BINDING_STAGE_VERTEX,
                    (__bridge void *)convertedBuffer, 0, (uint32_t)bindingIndex)) {
                mglBindingStateSetVertexBuffer(encCtx->encoder, convertedBuffer,
                                               0, bindingIndex);
                mglRenderCppBindingUpdateVertexBuffer(
                    _bindingStateOwner, (__bridge void *)convertedBuffer, 0,
                    (uint32_t)bindingIndex);
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

        /* Default: bind the VBO at offset 0; per-attribute offsets live in the
         * vertex descriptor (binding_offset + relativeoffset).
         * Absolute mode (BindNoFlush dynamic VAO batches): descriptor has only
         * relativeoffset, so pass VERTEX_BINDING_OFFSET here. */
        NSUInteger metalBindOffset =
            _batching.absoluteVertexBindingOffsets ? attribBindingOffset : 0u;
	    if (!mglBindingStateIsValid(_bindingStateOwner) ||
                !mglBindingStateBufferMatches(
                    _bindingStateOwner, MGL_RENDER_CPP_BINDING_STAGE_VERTEX,
                    (__bridge void *)attribMetalBuffer, metalBindOffset, (uint32_t)bindingIndex)) {
	        mglBindingStateSetVertexBuffer(encCtx->encoder, attribMetalBuffer,
	                                       metalBindOffset, bindingIndex);
	        mglRenderCppBindingUpdateVertexBuffer(
                    _bindingStateOwner, (__bridge void *)attribMetalBuffer, metalBindOffset,
                    (uint32_t)bindingIndex);
	        MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
	        mglNoteBufferEncoded(attribBuffer);
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
    const int vertexStage = _tessellation.nativeTESActive
        ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;

    if (!fallbackBindingBuffer) {
        fallbackBindingBuffer = mglBindingStateCreateBuffer(
            _device, kMGLDefaultStageFallbackBufferSize,
            MTLResourceStorageModeShared);
    }

    // Bind fallback buffer for required stage buffer bindings that were not mapped.
    // This prevents Metal validation aborts on missing buffer slots.
    const int resourceTypes[] = {
        _UNIFORM_BUFFER_RES,
        _UNIFORM_CONSTANT_RES,
        _STORAGE_BUFFER_RES,
        _ATOMIC_COUNTER_RES
    };
    for (int t = 0; t < 4; t++) {
        int resourceType = resourceTypes[t];
        int count = [self getProgramBindingCount:vertexStage type:resourceType];
        Program *program = activeProgram;
        for (int i = 0; i < count; i++) {
            if (!program || resourceType < 0 || resourceType >= _MAX_SPIRV_RES ||
                i >= (int)program->spirv_resources_list[vertexStage][resourceType].count) {
                continue;
            }
            SpirvResource *resource = &program->spirv_resources_list[vertexStage][resourceType].list[i];
            if (mglShouldSkipStageBufferResource(program, vertexStage, resourceType, resource)) {
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
                    if (!mglBindingStateIsValid(_bindingStateOwner) ||
                !mglBindingStateBufferMatches(
                    _bindingStateOwner, MGL_RENDER_CPP_BINDING_STAGE_VERTEX,
                    (__bridge void *)fallbackBindingBuffer, 0, (uint32_t)_slot)) {
                        mglBindingStateSetVertexBuffer(encCtx->encoder,
                                                       fallbackBindingBuffer,
                                                       0, _slot);
                        mglRenderCppBindingUpdateVertexBuffer(
                    _bindingStateOwner, (__bridge void *)fallbackBindingBuffer, 0,
                    (uint32_t)_slot);
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
                if (!mglBindingStateIsValid(_bindingStateOwner) ||
                !mglBindingStateBufferMatches(
                    _bindingStateOwner, MGL_RENDER_CPP_BINDING_STAGE_VERTEX,
                    (__bridge void *)fallbackBindingBuffer, 0, (uint32_t)s)) {
                    mglBindingStateSetVertexBuffer(encCtx->encoder,
                                                   fallbackBindingBuffer, 0, s);
                    mglRenderCppBindingUpdateVertexBuffer(
                    _bindingStateOwner, (__bridge void *)fallbackBindingBuffer, 0,
                    (uint32_t)s);
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
        if (!pointProgram) continue;
        if (pointProgram->uses_point_size_params) {
            needsPointSizeParams = YES;
            break;
        }
    }
    if (needsPointSizeParams) {
        float pointSizeParams[2] = {
            ctx && MGL_STATE(ctx)->var.point_size > 0.0f ? MGL_STATE(ctx)->var.point_size : 1.0f,
            ctx && MGL_STATE(ctx)->caps.program_point_size ? 1.0f : 0.0f
        };
        mglBindingStateSetVertexBytes(encCtx->encoder, pointSizeParams,
                                      sizeof(pointSizeParams),
                                      kMGLPointSizeParamBufferIndex);
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
    double fbindStartSeconds = mglTraceNowSeconds();
    uint64_t fbindStartNS = mglTraceClockNS();
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

    mapCount = MGL_STATE(ctx)->fragment_buffer_map_list.count;
    if (mapCount > MAX_MAPPED_BUFFERS) {
        static uint64_t s_fbindMapCountOverflow = 0;
        uint64_t hit = ++s_fbindMapCountOverflow;
        if (hit <= 16ull || (hit % 4096ull) == 0ull) {
            NSLog(@"MGL WARNING: FBIND mapCount=%u exceeds MAX_MAPPED_BUFFERS=%d, clamping (hit=%llu)",
                  mapCount, MAX_MAPPED_BUFFERS, (unsigned long long)hit);
        }
        mapCount = MAX_MAPPED_BUFFERS;
    }

    for (GLuint i = 0; i < mapCount; i++)
    {
        map = &MGL_STATE(ctx)->fragment_buffer_map_list.buffers[i];

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
            mglBindingStateSetFragmentBuffer(encCtx->encoder, nil, 0,
                                             bindingIndex);
            mglRenderCppBindingClearFragmentBuffer(_bindingStateOwner,
                                                         (uint32_t)bindingIndex);
            continue;
        }

        if (offset < 0) {
            NSLog(@"MGL FBIND skip slot=%u buffer=%u: negative offset=%lld",
                  i, ptr->name, (long long)offset);
            mglBindingStateSetFragmentBuffer(encCtx->encoder, nil, 0,
                                             bindingIndex);
            mglRenderCppBindingClearFragmentBuffer(_bindingStateOwner,
                                                         (uint32_t)bindingIndex);
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
                    mglBindingStateSetFragmentBuffer(encCtx->encoder, nil, 0,
                                                     bindingIndex);
                    mglRenderCppBindingClearFragmentBuffer(_bindingStateOwner,
                                                         (uint32_t)bindingIndex);
            mglRenderCppBindingClearFragmentBuffer(_bindingStateOwner,
                                                         (uint32_t)bindingIndex);
                    continue;
                }

                size_t bindOffset = (size_t)offset;
                size_t bufferSize = (size_t)ptr->size;
                if (bindOffset >= bufferSize) {
                    NSLog(@"MGL FBIND skip small buffer=%u slot=%u: offset=%lu bufferSize=%lu",
                          ptr->name, i, (unsigned long)bindOffset, (unsigned long)bufferSize);
                    mglBindingStateSetFragmentBuffer(encCtx->encoder, nil, 0,
                                                     bindingIndex);
                    mglRenderCppBindingClearFragmentBuffer(_bindingStateOwner,
                                                         (uint32_t)bindingIndex);
            mglRenderCppBindingClearFragmentBuffer(_bindingStateOwner,
                                                         (uint32_t)bindingIndex);
                    continue;
                }

                size_t bindLength = bufferSize - bindOffset;
                const uint8_t *bindPtr = ((const uint8_t *)ptr->data.buffer_data) + bindOffset;
                mglBindingStateSetFragmentBytes(encCtx->encoder, bindPtr,
                                                bindLength, bindingIndex);
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
                    mglBindingStateSetFragmentBuffer(encCtx->encoder, nil, 0,
                                                     bindingIndex);
                    mglRenderCppBindingClearFragmentBuffer(_bindingStateOwner,
                                                         (uint32_t)bindingIndex);
            mglRenderCppBindingClearFragmentBuffer(_bindingStateOwner,
                                                         (uint32_t)bindingIndex);
                    continue;
                }
                id<MTLBuffer> fallbackBuffer = (__bridge id<MTLBuffer>)(ptr->data.mtl_data);
                if (fallbackBuffer) {
                    NSUInteger metalLen = fallbackBuffer.length;
                    NSUInteger bindOffset = (NSUInteger)offset;
                    if (bindOffset >= metalLen) {
                        NSLog(@"MGL FBIND skip small MTL buffer=%u slot=%u: offset=%lu length=%lu",
                              ptr->name, i, (unsigned long)bindOffset, (unsigned long)metalLen);
                        mglBindingStateSetFragmentBuffer(encCtx->encoder, nil, 0,
                                                         bindingIndex);
                    mglRenderCppBindingClearFragmentBuffer(_bindingStateOwner,
                                                         (uint32_t)bindingIndex);
            mglRenderCppBindingClearFragmentBuffer(_bindingStateOwner,
                                                         (uint32_t)bindingIndex);
                        continue;
                    }

                    if (!mglBindingStateIsValid(_bindingStateOwner) ||
                !mglBindingStateBufferMatches(
                    _bindingStateOwner, MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT,
                    (__bridge void *)fallbackBuffer, (NSUInteger)offset, (uint32_t)bindingIndex)) {
                        mglBindingStateSetFragmentBuffer(
                            encCtx->encoder, fallbackBuffer,
                            (NSUInteger)offset, bindingIndex);
                        mglRenderCppBindingUpdateFragmentBuffer(
                    _bindingStateOwner, (__bridge void *)fallbackBuffer, (NSUInteger)offset,
                    (uint32_t)bindingIndex);
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
                mglBindingStateSetFragmentBuffer(encCtx->encoder, nil, 0,
                                                 bindingIndex);
            mglRenderCppBindingClearFragmentBuffer(_bindingStateOwner,
                                                         (uint32_t)bindingIndex);
            }
            
            // clear buffer data dirty bits
            ptr->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
        }
        else
        {
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

            /* For small uniform constants (plain uniforms), use setFragmentBytes
             * to copy the data into the command buffer at bind time. This is
             * critical for correctness when the same uniform buffer is updated
             * between draws encoded into the same command buffer — a shared-
             * memory MTLBuffer would let the GPU see only the final value.
             *
             * See the vertex-stage counterpart: taking this before
             * bindMTLBuffer keeps these slots free of an MTLBuffer, which
             * otherwise costs a copy-on-write snapshot per glUniform* upload
             * plus a zero-padded isolated buffer per draw. */
            if (isBaseBinding &&
                map->resource_type == _UNIFORM_CONSTANT_RES &&
                ptr->data.buffer_data &&
                offset == 0 &&
                requiredBindingBytes <= kMGLStageBindingStackScratchSize) {
                NSUInteger visibleBytes =
                    (NSUInteger)mglBufferMapVisibleBackingBytes(map, ptr->data.buffer_size);
                NSUInteger inlineLength = MAX(visibleBytes, requiredBindingBytes);
                if (visibleBytes > 0 && inlineLength <= kMGLStageBindingStackScratchSize) {
                    uint8_t padded[kMGLStageBindingStackScratchSize];
                    const void *inlineBytes = (const void *)(uintptr_t)ptr->data.buffer_data;
                    if (inlineLength > visibleBytes) {
                        memcpy(padded, inlineBytes, visibleBytes);
                        memset(padded + visibleBytes, 0, inlineLength - visibleBytes);
                        inlineBytes = padded;
                    }
                    mglBindingStateSetFragmentBytes(encCtx->encoder,
                                                    inlineBytes, inlineLength,
                                                    bindingIndex);
                    [self invalidateLastBoundFragmentBufferAtIndex:bindingIndex];
                    anyBindingPresent[bindingIndex] = true;
                    if (!ptr->data.mtl_data) {
                        ptr->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
                    }
                    if (kMGLVerboseBindLogs) {
                        NSLog(@"MGL FBIND uniform-constant setFragmentBytes slot=%lu buffer=%u len=%lu visible=%lu",
                              (unsigned long)bindingIndex,
                              ptr->name,
                              (unsigned long)inlineLength,
                              (unsigned long)visibleBytes);
                    }
                    continue;
                }
            }

            if (!ptr->data.mtl_data) {
                [self bindMTLBuffer:ptr];
            } else if (ptr->data.dirty_bits & (DIRTY_BUFFER_DATA | DIRTY_BUFFER_ADDR)) {
                /* Same first-draw refresh as the vertex path above. */
                [self updateDirtyBuffer:ptr];
            }
            id<MTLBuffer> buffer = nil;
            if (ptr->data.mtl_data &&
                (uintptr_t)ptr->data.mtl_data >= 0x100000000ULL) {
                buffer = (__bridge id<MTLBuffer>)(ptr->data.mtl_data);
            }

            NSUInteger metalLen = buffer ? buffer.length : 0u;
            NSUInteger availableBytes = buffer
                ? mglBufferMapVisibleBackingBytes(map, metalLen)
                : 0u;

            if (!ptr->gpu_write_target &&
                (!buffer || bindOffset >= metalLen ||
                 availableBytes < requiredBindingBytes)) {
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
                    mglBindingStateSetFragmentBuffer(encCtx->encoder, nil, 0,
                                                     bindingIndex);
                    mglRenderCppBindingClearFragmentBuffer(_bindingStateOwner,
                                                         (uint32_t)bindingIndex);
                    continue;
                }

                mglBindingStateSetFragmentBuffer(encCtx->encoder, isolated, 0,
                                                 bindingIndex);
                mglRenderCppBindingUpdateFragmentBuffer(
                    _bindingStateOwner, (__bridge void *)isolated, 0,
                    (uint32_t)bindingIndex);
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
            
            if (!mglBindingStateIsValid(_bindingStateOwner) ||
                !mglBindingStateBufferMatches(
                    _bindingStateOwner, MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT,
                    (__bridge void *)buffer, (NSUInteger)offset, (uint32_t)bindingIndex)) {
                mglBindingStateSetFragmentBuffer(encCtx->encoder, buffer,
                                                 (NSUInteger)offset,
                                                 bindingIndex);
                mglRenderCppBindingUpdateFragmentBuffer(
                    _bindingStateOwner, (__bridge void *)buffer, (NSUInteger)offset,
                    (uint32_t)bindingIndex);
                MGL_PERF_INC(g_mglSetFragmentBufferCallsSinceSwap);
                mglNoteBufferEncoded(ptr);
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
    mglRenderCppBindingOrFragmentBufferMask(_bindingStateOwner, boundFragmentBufferMask);

    if (mglEnvFlagEnabled("MGL_TRACE_SPARSE_BINDING")) {
        static uint64_t s_fbind_trace_count = 0;
        if ((++s_fbind_trace_count % 500) == 1) {
            int textureSlotCount =
                mglBindingStateTextureSlotCount(_bindingStateOwner);
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
        mglTraceLogNSString(@"MGL TRACE fbind.end call=%llu mapCount=%u boundSlots=%lu baseSlots=%lu elapsed=%.1fus",
              (unsigned long long)fbindCall,
              (unsigned)mapCount,
              (unsigned long)boundSlots,
              (unsigned long)baseSlots,
              (mglTraceClockNS() - fbindStartNS) / 1000.0);
    }

    /* Mark the dedup cache as valid for the current encoder so subsequent
     * binds can be skipped when the resource and offset are unchanged. */
    mglRenderCppBindingSetValid(_bindingStateOwner, 1);
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
        fallbackBindingBuffer = mglBindingStateCreateBuffer(
            _device, kMGLDefaultStageFallbackBufferSize,
            MTLResourceStorageModeShared);
    }

    // Bind fallback buffer for required stage buffer bindings that were not mapped.
    const int resourceTypes[] = {
        _UNIFORM_BUFFER_RES,
        _UNIFORM_CONSTANT_RES,
        _STORAGE_BUFFER_RES,
        _ATOMIC_COUNTER_RES
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
                    if (!mglBindingStateIsValid(_bindingStateOwner) ||
                !mglBindingStateBufferMatches(
                    _bindingStateOwner, MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT,
                    (__bridge void *)fallbackBindingBuffer, 0, (uint32_t)_slot)) {
                        mglBindingStateSetFragmentBuffer(
                            encCtx->encoder, fallbackBindingBuffer, 0, _slot);
                        mglRenderCppBindingUpdateFragmentBuffer(
                    _bindingStateOwner, (__bridge void *)fallbackBindingBuffer, 0,
                    (uint32_t)_slot);
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
                if (!mglBindingStateIsValid(_bindingStateOwner) ||
                !mglBindingStateBufferMatches(
                    _bindingStateOwner, MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT,
                    (__bridge void *)fallbackBindingBuffer, 0, (uint32_t)s)) {
                    mglBindingStateSetFragmentBuffer(
                        encCtx->encoder, fallbackBindingBuffer, 0, s);
                    mglRenderCppBindingUpdateFragmentBuffer(
                    _bindingStateOwner, (__bridge void *)fallbackBindingBuffer, 0,
                    (uint32_t)s);
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
        /* Full clear when trace is active — trace consumers read all fields. */
        memset(_resourceFallback.fragmentTextureTraceBindings, 0,
               sizeof(_resourceFallback.fragmentTextureTraceBindings));
    } else {
        /* When trace is disabled, only clear the functional flag fields read
         * by non-trace consumers (~384 bytes vs ~12 KB). */
        mglClearFragmentTextureTraceFunctionalFlags(
            _resourceFallback.fragmentTextureTraceBindings, TEXTURE_UNITS);
    }


    const int vertexResourceStage = _tessellation.nativeTESActive
        ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;
    vertexProgram = _tessellation.nativeTESActive
        ? _tessellation.nativeTESProgram
        : mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    fragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    vertexProgramName = vertexProgram ? vertexProgram->name : mglCurrentRenderProgramKey(ctx);
    fragmentProgramName = fragmentProgram ? fragmentProgram->name : mglCurrentRenderProgramKey(ctx);

    id<MTLSamplerState> defaultSampler = [self fallbackSamplerState];
    if (defaultSampler) {
        /* Only warmup sampler slots the program actually samples, using the
         * sampled_texture_unit_mask bitmap to skip unused slots, instead of
         * blindly setting all TEXTURE_UNITS. */
        uint32_t activeMask[4] = {0, 0, 0, 0};
        if (vertexProgram) {
            (void)mglProgramSamplesTextureUnit(vertexProgram, 0); /* trigger lazy build */
            for (int i = 0; i < 4; i++)
                activeMask[i] |= vertexProgram->sampled_texture_unit_mask[i];
        }
        if (fragmentProgram && fragmentProgram != vertexProgram) {
            (void)mglProgramSamplesTextureUnit(fragmentProgram, 0);
            for (int i = 0; i < 4; i++)
                activeMask[i] |= fragmentProgram->sampled_texture_unit_mask[i];
        }

        NSUInteger warmupCount = TEXTURE_UNITS;
        if (warmupCount > kMaxFragmentSamplerSlots) {
            warmupCount = kMaxFragmentSamplerSlots;
        }
        /* If no program is bound (both NULL), fall back to warming all slots
         * to avoid Metal assertions on stale sampler state. */
        bool hasActiveProgram = (vertexProgram != nil || fragmentProgram != nil);
        bool maskEmpty = (activeMask[0] | activeMask[1] |
                          activeMask[2] | activeMask[3]) == 0u;
        if (hasActiveProgram && !maskEmpty) {
            for (NSUInteger s = 0; s < warmupCount; s++) {
                if ((activeMask[s >> 5] & (1u << (s & 31u))) == 0u) continue;
                [self setVertexSamplerStateIfNeeded:defaultSampler atIndex:s];
                [self setFragmentSamplerStateIfNeeded:defaultSampler atIndex:s];
            }
        } else {
            for (NSUInteger s = 0; s < warmupCount; s++) {
                [self setVertexSamplerStateIfNeeded:defaultSampler atIndex:s];
                [self setFragmentSamplerStateIfNeeded:defaultSampler atIndex:s];
            }
        }
    }


    GLuint sampledCount = 0;
    GLuint separateSamplerCount = 0;
    GLuint boundSeparateSamplers = 0;

    // Metal validates every active stage resource. Bind vertex-stage sampled
    // images as well, even though most Minecraft pipelines only sample in FS.
    vertexSampledCount = [self getProgramBindingCount:vertexResourceStage
                                                  type:_SAMPLED_IMAGE_RES];
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
        mglTraceLogNSString(@"MGL TRACE texbind.summary call=%llu program=%u vertexSampled=%u vertexBoundTex=%u vertexFallback=%u sampled=%u boundTex=%u nilTex=%u fallbackTex=%u sampledSamplers=%u separateSamplers=%u boundSeparate=%u",
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
    const int vertexStage = _tessellation.nativeTESActive
        ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;

    for (GLuint i = 0; i < vertexSampledCount; i++)
    {
        Program *currentProgram = vertexProgram;
        SpirvResource *sampledResource = NULL;
        const char *sampledName = "";
        if (currentProgram &&
            i < currentProgram->spirv_resources_list[vertexStage][_SAMPLED_IMAGE_RES].count) {
            sampledResource = &currentProgram->spirv_resources_list[vertexStage][_SAMPLED_IMAGE_RES].list[i];
            sampledName = sampledResource->name;
        }
        /* read binding/gl_binding directly from the already-resolved
         * SpirvResource instead of re-resolving the program per query. When
         * sampledResource is NULL (no program / index OOR), mirror the
         * query-method semantics of returning 0. */
        GLuint spirvBinding = sampledResource ? sampledResource->binding : 0u;
        GLuint glBinding = sampledResource ? sampledResource->gl_binding : 0u;
        if (spirvBinding >= TEXTURE_UNITS || glBinding >= TEXTURE_UNITS) {
            continue;
        }
        if (mglShouldSkipStageTextureResource(currentProgram,
                                              vertexStage,
                                              _SAMPLED_IMAGE_RES,
                                              sampledResource)) {
            continue;
        }
        GLuint textureUnit = [self textureUnitForSampledResource:sampledResource
                                                        program:currentProgram
                                                    metalBinding:spirvBinding
                                                           stage:vertexStage];
        /* derive texture types/data kind directly from sampledResource
         * via C helpers, skipping per-resource mglResolveProgramForStageFromState. */
        MTLTextureType expectedType = mglExpectedTextureTypeForResource(currentProgram, vertexStage, sampledResource);
        MTLTextureType lookupType = mglDeclaredTextureTypeFromResource(sampledResource);
        MGLTextureDataKind expectedKind = mglExpectedTextureDataKindForResource(currentProgram, vertexStage, sampledResource);
        Texture *ptr = [self textureForSampledResource:sampledResource
                                          metalBinding:spirvBinding
                                                  stage:vertexStage
                                           expectedType:(lookupType ? lookupType : expectedType)
                                          textureUnit:textureUnit];
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

            if (textureUnit < TEXTURE_UNITS && MGL_STATE(ctx)->texture_samplers[textureUnit]) {
                Sampler *glSampler = MGL_STATE(ctx)->texture_samplers[textureUnit];
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
        GLuint samplerBinding = sampledResource && sampledResource->has_combined_sampler
            ? mglMetalCombinedSamplerSlot(sampledResource)
            : spirvBinding;
        if (sampler &&
            (!sampledResource || sampledResource->has_combined_sampler) &&
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
            Texture *unitActive = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->active_textures[textureUnit] : NULL;
            Texture *unitExpected = (textureUnit < TEXTURE_UNITS &&
                                     expectedIndex >= 0 &&
                                     expectedIndex < _MAX_TEXTURE_TYPES)
                ? MGL_STATE(ctx)->texture_units[textureUnit].textures[expectedIndex]
                : NULL;
            Texture *unit2D = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_2D] : NULL;
            Texture *unitCube = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_CUBE_MAP] : NULL;
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
            if (hit <= 8ull || (hit % 2048ull) == 0ull) {
                Texture *unitActive = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->active_textures[textureUnit] : NULL;
                Texture *unitBuffer = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_BUFFER_TARGET] : NULL;
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
                    Texture *unitActive = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->active_textures[textureUnit] : NULL;
                    Texture *unitExpected = (expectedIndex >= 0 && expectedIndex < _MAX_TEXTURE_TYPES)
                        ? MGL_STATE(ctx)->texture_units[textureUnit].textures[expectedIndex]
                        : NULL;
                    uint64_t levelDataHash = (sampleLevel0 && sampleLevel0->data && sampleLevel0->data_size > 0)
                        ? mglTraceHashBytes((const void *)(uintptr_t)sampleLevel0->data, sampleLevel0->data_size)
                        : 0ull;

                    mglTraceLogNSString(@"MGL TRACE texbind.sample-detail call=%llu hit=%llu stage=vertex program=%u name=%s binding=%u "
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
    *sampledCount = [self getProgramBindingCount:_FRAGMENT_SHADER type:_SAMPLED_IMAGE_RES];
    for (GLuint i = 0; i < *sampledCount; i++)
    {
        Program *sampleProgram = fragmentProgram;
        SpirvResource *sampledResource = NULL;
        const char *sampledName = "";
        if (sampleProgram &&
            i < sampleProgram->spirv_resources_list[_FRAGMENT_SHADER][_SAMPLED_IMAGE_RES].count) {
            sampledResource = &sampleProgram->spirv_resources_list[_FRAGMENT_SHADER][_SAMPLED_IMAGE_RES].list[i];
            sampledName = sampledResource->name;
        }
        /* read binding/gl_binding directly from the already-resolved
         * SpirvResource instead of re-resolving the program per query. */
        GLuint spirvBinding = sampledResource ? sampledResource->binding : 0u;
        GLuint glBinding = sampledResource ? sampledResource->gl_binding : 0u;
        if (spirvBinding >= TEXTURE_UNITS || glBinding >= TEXTURE_UNITS) {
            continue;
        }
        if (mglShouldSkipStageTextureResource(sampleProgram,
                                              _FRAGMENT_SHADER,
                                              _SAMPLED_IMAGE_RES,
                                              sampledResource)) {
            continue;
        }
        GLuint textureUnit = [self textureUnitForSampledResource:sampledResource
                                                        program:sampleProgram
                                                    metalBinding:spirvBinding
                                                           stage:_FRAGMENT_SHADER];

        /* derive texture types/data kind directly from sampledResource
         * via C helpers, skipping per-resource mglResolveProgramForStageFromState. */
        MTLTextureType expectedType = mglExpectedTextureTypeForResource(sampleProgram, _FRAGMENT_SHADER, sampledResource);
        MTLTextureType lookupType = mglDeclaredTextureTypeFromResource(sampledResource);
        MGLTextureDataKind expectedKind = mglExpectedTextureDataKindForResource(sampleProgram, _FRAGMENT_SHADER, sampledResource);
        Texture *ptr = [self textureForSampledResource:sampledResource
                                          metalBinding:spirvBinding
                                                  stage:_FRAGMENT_SHADER
                                           expectedType:(lookupType ? lookupType : expectedType)
                                          textureUnit:textureUnit];
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
	                Texture *unitActive = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->active_textures[textureUnit] : NULL;
	                Texture *unitExpected = (expectedIndex >= 0 && expectedIndex < _MAX_TEXTURE_TYPES)
	                    ? MGL_STATE(ctx)->texture_units[textureUnit].textures[expectedIndex]
	                    : NULL;
	                Texture *unit2D = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_2D] : NULL;
	                Texture *unitCube = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_CUBE_MAP] : NULL;
	                MTLTextureType actualType = texture ? texture.textureType : 0;
	                uint64_t levelDataHash = (sampleLevel0 && sampleLevel0->data && sampleLevel0->data_size > 0)
	                    ? mglTraceHashBytes((const void *)(uintptr_t)sampleLevel0->data, sampleLevel0->data_size)
	                    : 0ull;

	                mglTraceLogNSString(@"MGL TRACE texbind.sample-detail call=%llu hit=%llu stage=fragment program=%u name=%s binding=%u "
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
                            Texture *atlasUnitActive = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->active_textures[textureUnit] : NULL;
                            Texture *atlasUnitExpected = (atlasExpectedIndex >= 0 && atlasExpectedIndex < _MAX_TEXTURE_TYPES)
                                ? MGL_STATE(ctx)->texture_units[textureUnit].textures[atlasExpectedIndex]
                                : NULL;
                            Texture *atlasUnit2D = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_2D] : NULL;
                            Texture *atlasUnitCube = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_CUBE_MAP] : NULL;
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
                                    (unsigned)(ctx ? MGL_STATE(ctx)->program_name : 0u),
                                    (unsigned)(ctx ? MGL_STATE(ctx)->program_name : 0u),
                                    (unsigned)(ctx ? MGL_STATE(ctx)->var.program_pipeline_binding : 0u),
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
                                    (unsigned)(ctx && MGL_STATE(ctx)->framebuffer ? MGL_STATE(ctx)->framebuffer->name : 0u),
                                    (unsigned)_renderPassManager.state->renderPassFramebufferName,
                                    rpColor0,
                                    rpDepth,
                                    ctx && MGL_STATE(ctx)->caps.depth_test ? 1 : 0,
                                    ctx && MGL_STATE(ctx)->caps.blend ? 1 : 0);
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
            Texture *unitActive = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->active_textures[textureUnit] : NULL;
            Texture *unitExpected = (textureUnit < TEXTURE_UNITS &&
                                     expectedIndex >= 0 &&
                                     expectedIndex < _MAX_TEXTURE_TYPES)
                ? MGL_STATE(ctx)->texture_units[textureUnit].textures[expectedIndex]
                : NULL;
            Texture *unit2D = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_2D] : NULL;
            Texture *unitCube = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_CUBE_MAP] : NULL;
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
        GLuint samplerBinding = sampledResource && sampledResource->has_combined_sampler
            ? mglMetalCombinedSamplerSlot(sampledResource)
            : spirvBinding;
        if (sampler &&
            (!sampledResource || sampledResource->has_combined_sampler) &&
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

            mglTraceLogNSString(@"MGL TRACE texbind.sampled call=%llu idx=%u binding=%u glTex=%u target=0x%x internal=0x%x "
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
                    MGL_STATE(ctx)->recent_sampled_2d_textures[textureUnit][historyIndex];
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
        Texture *unitActive = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->active_textures[textureUnit] : NULL;
        Texture *unit2D = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_2D] : NULL;
        Texture *last2D = textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->last_sampled_2d_textures[textureUnit] : NULL;
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
                Framebuffer *currentFbo = ctx ? MGL_STATE(ctx)->framebuffer : NULL;
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
                        (unsigned)(ctx ? MGL_STATE(ctx)->program_name : 0u),
                        (unsigned)(ctx ? MGL_STATE(ctx)->program_name : 0u),
                        (unsigned)(ctx ? MGL_STATE(ctx)->var.program_pipeline_binding : 0u),
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
                                    (unsigned)(ctx ? MGL_STATE(ctx)->program_name : 0u),
                                    (unsigned)(ctx ? MGL_STATE(ctx)->program_name : 0u),
                                    (unsigned)(ctx ? MGL_STATE(ctx)->var.program_pipeline_binding : 0u),
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
                            (unsigned)(ctx ? MGL_STATE(ctx)->program_name : 0u),
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

    if (textureUnit < TEXTURE_UNITS && MGL_STATE(ctx)->texture_samplers[textureUnit]) {
        Sampler *glSampler = MGL_STATE(ctx)->texture_samplers[textureUnit];
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

    if (mglMipDiagEnabled() && ptr) {
        Sampler *glSampler = (textureUnit < TEXTURE_UNITS)
            ? MGL_STATE(ctx)->texture_samplers[textureUnit] : NULL;
        const TextureParameter *effective = glSampler ? &glSampler->params : &ptr->params;
        uint64_t signature = 1469598103934665603ULL;
        signature = mglMipDiagMixState(signature, ptr->name);
        signature = mglMipDiagMixState(signature, effective->min_filter);
        signature = mglMipDiagMixState(signature, effective->mag_filter);
        signature = mglMipDiagMixState(signature, ptr->params.base_level);
        signature = mglMipDiagMixState(signature, ptr->params.max_level);
        signature = mglMipDiagMixState(signature, texture ? texture.mipmapLevelCount : 0u);
        signature = mglMipDiagMixState(signature, (uint64_t)(uintptr_t)texture);
        /* A render-target atlas is sampled through the Y-flip copy, so a mip
         * level left dirty or a version mismatch is what a stale mip looks like. */
        signature = mglMipDiagMixState(signature, usedSampledCopyForTrace ? 1u : 0u);
        signature = mglMipDiagMixState(signature, ptr->mtl_gl_sampled_levels);
        signature = mglMipDiagMixState(signature, ptr->mtl_gl_sampled_dirty_mip_mask);
        signature = mglMipDiagMixState(signature,
            (uint64_t)(ptr->mtl_gl_sampled_write_version != ptr->mtl_render_target_write_version));

        static uint64_t s_fragSamplerState[TEXTURE_UNITS];
        if (textureUnit < TEXTURE_UNITS &&
            mglMipDiagStateChanged(&s_fragSamplerState[textureUnit], signature)) {
            NSLog(@"MGL MIP_DIAG frag unit=%u binding=%u program=%u glTex=%u "
                  @"source=%s minFilter=0x%x magFilter=0x%x minLod=%.1f maxLod=%.1f aniso=%.1f "
                  @"base=%u max=%u glLevels=%u mtlLevels=%lu mtlTex=%p "
                  @"renderTarget=%d viaCopy=%d copyLevels=%u dirtyMips=0x%x rtVer=%u copyVer=%u",
                  (unsigned)textureUnit,
                  (unsigned)spirvBinding,
                  (unsigned)fragmentProgramName,
                  (unsigned)ptr->name,
                  glSampler ? "glSampler" : "texParams",
                  (unsigned)effective->min_filter,
                  (unsigned)effective->mag_filter,
                  (double)effective->min_lod,
                  (double)effective->max_lod,
                  (double)effective->max_anisotropy,
                  (unsigned)ptr->params.base_level,
                  (unsigned)ptr->params.max_level,
                  (unsigned)ptr->num_levels,
                  (unsigned long)(texture ? texture.mipmapLevelCount : 0u),
                  texture,
                  ptr->is_render_target ? 1 : 0,
                  usedSampledCopyForTrace ? 1 : 0,
                  (unsigned)ptr->mtl_gl_sampled_levels,
                  (unsigned)ptr->mtl_gl_sampled_dirty_mip_mask,
                  (unsigned)ptr->mtl_render_target_write_version,
                  (unsigned)ptr->mtl_gl_sampled_write_version);
        }
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
    const int vertexStage = _tessellation.nativeTESActive
        ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;
    GLuint vertexStorageImageCount = [self getProgramBindingCount:vertexStage
                                                            type:_STORAGE_IMAGE_RES];
    for (GLuint i = 0; i < vertexStorageImageCount; i++)
    {
        SpirvResource *resource = NULL;
        if (vertexProgram &&
            i < vertexProgram->spirv_resources_list[vertexStage][_STORAGE_IMAGE_RES].count) {
            resource = &vertexProgram->spirv_resources_list[vertexStage][_STORAGE_IMAGE_RES].list[i];
        }
        if (mglShouldSkipStageTextureResource(vertexProgram,
                                              vertexStage,
                                              _STORAGE_IMAGE_RES,
                                              resource)) {
            continue;
        }
        GLuint glUnit = resource ? (resource->sampler_unit >= 0 ? (GLuint)resource->sampler_unit : resource->gl_binding)
                                 : [self getProgramGLBinding:vertexStage
                                                        type:_STORAGE_IMAGE_RES
                                                       index:(int)i];
        if (glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = MGL_STATE(ctx)->image_units[glUnit].tex;
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
            i < vertexProgram->spirv_resources_list[vertexStage][_STORAGE_IMAGE_RES].count) {
            resource = &vertexProgram->spirv_resources_list[vertexStage][_STORAGE_IMAGE_RES].list[i];
        }
        if (mglShouldSkipStageTextureResource(vertexProgram,
                                              vertexStage,
                                              _STORAGE_IMAGE_RES,
                                              resource)) {
            continue;
        }
        GLuint metalSlot = resource ? mglMetalResourceSlot(resource)
                                    : [self getProgramBinding:vertexStage
                                                        type:_STORAGE_IMAGE_RES
                                                       index:(int)i];
        GLuint glUnit = resource ? (resource->sampler_unit >= 0 ? (GLuint)resource->sampler_unit : resource->gl_binding)
                                 : [self getProgramGLBinding:vertexStage
                                                        type:_STORAGE_IMAGE_RES
                                                       index:(int)i];
        if (metalSlot >= TEXTURE_UNITS || glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = MGL_STATE(ctx)->image_units[glUnit].tex;
        id<MTLTexture> texture = nil;
        if (ptr) {
            MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
            texture = (__bridge id<MTLTexture>)(ptr->mtl_data);
            GLuint imgLevel = MGL_STATE(ctx)->image_units[glUnit].level;
            if (imgLevel > 0u && texture) {
                NSUInteger sliceCount = texture.arrayLength;
                if (texture.textureType == MTLTextureTypeCube ||
                    texture.textureType == MTLTextureTypeCubeArray) {
                    sliceCount = texture.arrayLength * 6u;
                }
                id<MTLTexture> levelView =
                    mglBindingStateCreateTextureLevelView(
                        texture, imgLevel, sliceCount);
                if (levelView) {
                    texture = levelView;
                }
            }
        }
        [self setVertexTextureIfNeeded:texture atIndex:metalSlot];
    }

    GLuint fragmentStorageImageCount = [self getProgramBindingCount:_FRAGMENT_SHADER
                                                               type:_STORAGE_IMAGE_RES];
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
            i < fragmentProgram->spirv_resources_list[_FRAGMENT_SHADER][_STORAGE_IMAGE_RES].count) {
            resource = &fragmentProgram->spirv_resources_list[_FRAGMENT_SHADER][_STORAGE_IMAGE_RES].list[i];
        }
        if (mglShouldSkipStageTextureResource(fragmentProgram,
                                              _FRAGMENT_SHADER,
                                              _STORAGE_IMAGE_RES,
                                              resource)) {
            continue;
        }

        GLuint glUnit = resource ? (resource->sampler_unit >= 0 ? (GLuint)resource->sampler_unit : resource->gl_binding)
                                 : [self getProgramGLBinding:_FRAGMENT_SHADER
                                                        type:_STORAGE_IMAGE_RES
                                                       index:(int)i];
        if (glUnit >= TEXTURE_UNITS) {
            continue;
        }

        Texture *ptr = MGL_STATE(ctx)->image_units[glUnit].tex;
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
            i < fragmentProgram->spirv_resources_list[_FRAGMENT_SHADER][_STORAGE_IMAGE_RES].count) {
            resource = &fragmentProgram->spirv_resources_list[_FRAGMENT_SHADER][_STORAGE_IMAGE_RES].list[i];
        }
        if (mglShouldSkipStageTextureResource(fragmentProgram,
                                              _FRAGMENT_SHADER,
                                              _STORAGE_IMAGE_RES,
                                              resource)) {
            continue;
        }

        GLuint metalSlot = resource ? mglMetalResourceSlot(resource)
                                    : [self getProgramBinding:_FRAGMENT_SHADER
                                                        type:_STORAGE_IMAGE_RES
                                                       index:(int)i];
        GLuint glUnit = resource ? (resource->sampler_unit >= 0 ? (GLuint)resource->sampler_unit : resource->gl_binding)
                                 : [self getProgramGLBinding:_FRAGMENT_SHADER
                                                        type:_STORAGE_IMAGE_RES
                                                       index:(int)i];
        if (metalSlot >= TEXTURE_UNITS || glUnit >= TEXTURE_UNITS) {
            continue;
        }

        Texture *ptr = MGL_STATE(ctx)->image_units[glUnit].tex;
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
            GLuint imgLevel = MGL_STATE(ctx)->image_units[glUnit].level;
            if (imgLevel > 0u && texture) {
                /* Cube and cube-array textures pack 6 face-slices per cube;
                 * the view's slice count must be a multiple of 6 for these
                 * types.  Other types use arrayLength directly. */
                NSUInteger sliceCount = texture.arrayLength;
                if (texture.textureType == MTLTextureTypeCube ||
                    texture.textureType == MTLTextureTypeCubeArray) {
                    sliceCount = texture.arrayLength * 6u;
                }
                id<MTLTexture> levelView =
                    mglBindingStateCreateTextureLevelView(
                        texture, imgLevel, sliceCount);
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
    *separateSamplerCount = [self getProgramBindingCount:_FRAGMENT_SHADER type:_SEPARATE_SAMPLERS_RES];
    *boundSeparateSamplers = 0;
    for (GLuint i = 0; i < *separateSamplerCount; i++)
    {
        GLuint spirvBinding = [self getProgramBinding:_FRAGMENT_SHADER type:_SEPARATE_SAMPLERS_RES index:(int)i];
        GLuint glBinding = [self getProgramGLBinding:_FRAGMENT_SHADER type:_SEPARATE_SAMPLERS_RES index:(int)i];
        if (spirvBinding >= TEXTURE_UNITS || glBinding >= TEXTURE_UNITS) {
            continue;
        }
        Program *sampleProgram = fragmentProgram;
        SpirvResource *samplerResource = NULL;
        if (sampleProgram &&
            i < sampleProgram->spirv_resources_list[_FRAGMENT_SHADER][_SEPARATE_SAMPLERS_RES].count) {
            samplerResource = &sampleProgram->spirv_resources_list[_FRAGMENT_SHADER][_SEPARATE_SAMPLERS_RES].list[i];
        }
        if (mglShouldSkipStageSamplerResource(sampleProgram,
                                              _FRAGMENT_SHADER,
                                              _SEPARATE_SAMPLERS_RES,
                                              samplerResource)) {
            continue;
        }
        GLuint textureUnit = [self textureUnitForSampledResource:samplerResource
                                                    metalBinding:spirvBinding
                                                           stage:_FRAGMENT_SHADER];

        id<MTLSamplerState> sampler = nil;
        if (textureUnit < TEXTURE_UNITS && MGL_STATE(ctx)->texture_samplers[textureUnit]) {
            Sampler *glSampler = MGL_STATE(ctx)->texture_samplers[textureUnit];
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
            mglTraceLogNSString(@"MGL TRACE texbind.separateSampler call=%llu idx=%u binding=%u unit=%u sampler=%p",
                  (unsigned long long)bindCall,
                  (unsigned)i,
                  (unsigned)spirvBinding,
                  (unsigned)textureUnit,
                  sampler);
        }
    }

    Program *arrayPrograms[] = { vertexProgram, fragmentProgram };
    int arrayStages[] = {
        _tessellation.nativeTESActive
            ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER,
        _FRAGMENT_SHADER
    };
    for (NSUInteger programIndex = 0; programIndex < 2; programIndex++) {
        Program *arrayProgram = arrayPrograms[programIndex];
        int arrayStage = arrayStages[programIndex];
        if (!arrayProgram) {
            continue;
        }

        SpirvResourceList *arrayResources =
            &arrayProgram->spirv_resources_list[arrayStage][_SAMPLED_IMAGE_RES];
        for (GLuint resourceIndex = 0; arrayResources->list && resourceIndex < arrayResources->count; resourceIndex++) {
            SpirvResource *resource = &arrayResources->list[resourceIndex];
            if (resource->gl_array_size <= 1) {
                continue;
            }

            MTLTextureType expectedType = [self getProgramExpectedTextureType:arrayStage
                                                                         type:_SAMPLED_IMAGE_RES
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
                    if (textureUnit < TEXTURE_UNITS && MGL_STATE(ctx)->texture_samplers[textureUnit]) {
                        Sampler *glSampler = MGL_STATE(ctx)->texture_samplers[textureUnit];
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
                    if (resource->has_combined_sampler && metalSampler &&
                        samplerSlot < kMaxFragmentSamplerSlots) {
                        [self setVertexSamplerStateIfNeeded:metalSampler atIndex:samplerSlot];
                    }
                } else {
                    [self setFragmentTextureIfNeeded:metalTexture atIndex:metalSlot];
                    if (resource->has_combined_sampler && metalSampler &&
                        samplerSlot < kMaxFragmentSamplerSlots) {
                        [self setFragmentSamplerStateIfNeeded:metalSampler atIndex:samplerSlot];
                    }
                }
            }
        }
    }
}




@end
