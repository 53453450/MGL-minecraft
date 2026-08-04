// MGLRenderer+Tessellation.m
// Tessellation compute path (TCS/TES dispatch) extracted from MGLRenderer.m.
// GL_PATCHES draws run as consecutive Metal compute encoders: the TCS kernel
// writes per-patch output plus tess factors, then the TES kernel consumes them.

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Tessellation_Private.h"
#import "mgl_sampler_compat.h"
#import "mgl_trace_log.h"
#import "mgl_msl_compiler.h"
#import "mgl_compute_pipeline_cache.h"
#import "mgl_metal_bridge.h"
#import "msl_patch_pipeline.h"

extern void mglRecordActivePrimitiveQueryDraw(GLMContext ctx, GLuint64 generated, GLuint64 written);

static const uint8_t *mglRendererReadableBufferBytes(Buffer *buffer)
{
    if (!buffer) {
        return NULL;
    }
    if (buffer->data.buffer_data && ((uintptr_t)buffer->data.buffer_data >= 0x1000ull)) {
        return (const uint8_t *)(uintptr_t)buffer->data.buffer_data;
    }
    if (buffer->data.mtl_data) {
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)(buffer->data.mtl_data);
        if (mtlBuffer && mtlBuffer.contents) {
            return (const uint8_t *)mtlBuffer.contents;
        }
    }
    return NULL;
}

@implementation MGLRenderer (Tessellation)

typedef struct {
    id<MTLBuffer> __strong buffer;
    NSUInteger offset;
    id<MTLBuffer> __strong initialization_source;
    NSUInteger initialization_source_offset;
    NSUInteger initialization_length;
    BOOL valid;
} MGLTessStageBufferBinding;

typedef struct {
    MGLTessStageBufferBinding slots[kMGLMaxBufferSlots];
    id<MTLBuffer> __strong size_buffer;
} MGLTessStageBufferBindingList;

/* Tessellation shaders run as consecutive compute encoders. Prepare their
 * buffer bindings before opening the next encoder so an isolated binding can
 * be initialized by an ordered GPU copy from a buffer written by the previous
 * stage. Reading source.contents here would capture stale CPU bytes while the
 * preceding TCS encoder is still pending on the same command buffer. */
- (bool)prepareTessStageBufferBindings:(MGLTessStageBufferBindingList *)bindings
                                 stage:(int)stage
                             copyBacks:(MGLStageBindingCopyBackList *)copyBacks
{
    if (!bindings || !copyBacks) {
        return false;
    }

    BufferMapList stageBufferMap = {0};
    if (![self mapGLBuffersToMTLBufferMap:&stageBufferMap stage:stage]) {
        return false;
    }

    /* Complete every lazy allocation before creating the initialization blit
     * encoder. bindMTLBuffer: may itself need an encoder. */
    for (GLuint i = 0; i < stageBufferMap.count; i++) {
        Buffer *ptr = stageBufferMap.buffers[i].buf;
        if (ptr && !ptr->data.mtl_data) {
            [self bindMTLBuffer:ptr];
        }
    }

    for (GLuint i = 0; i < stageBufferMap.count; i++) {
        BufferMap *map = &stageBufferMap.buffers[i];
        Buffer *ptr = map->buf;
        if (!ptr) {
            continue;
        }

        NSUInteger metalBindingIndex = map->has_metal_binding
            ? (NSUInteger)map->metal_binding_index
            : (NSUInteger)map->buffer_base_index;
        if (metalBindingIndex >= kMGLMaxMetalVertexBufferCount) {
            continue;
        }
        [self clearStageBindingCopyBack:copyBacks atIndex:metalBindingIndex];
        if (map->offset < 0) {
            return false;
        }
        NSUInteger bindOffset = (NSUInteger)map->offset;

        id<MTLBuffer> buffer = ptr->data.mtl_data
            ? (__bridge id<MTLBuffer>)(ptr->data.mtl_data)
            : nil;
        NSUInteger requiredBytes =
            [self getProgramBindingRequiredSize:stage
                                           type:(int)map->resource_type
                                          index:(int)map->resource_index];
        if (map->resource_type == SPVC_RESOURCE_TYPE_ATOMIC_COUNTER &&
            requiredBytes < sizeof(uint32_t)) {
            requiredBytes = sizeof(uint32_t);
        }

        GLsizeiptr storageRemaining = mglBufferMapStorageRemaining(map);
        NSUInteger availableBytes = buffer
            ? mglBufferMapVisibleBackingBytes(map, buffer.length)
            : 0u;
        BOOL needsIsolatedBinding =
            !buffer ||
            storageRemaining <= 0 ||
            bindOffset >= buffer.length ||
            availableBytes == 0 ||
            (requiredBytes > 0 && availableBytes < requiredBytes);

        MGLTessStageBufferBinding *binding = &bindings->slots[metalBindingIndex];
        binding->buffer = nil;
        binding->offset = 0u;
        binding->initialization_source = nil;
        binding->initialization_source_offset = 0u;
        binding->initialization_length = 0u;
        binding->valid = YES;
        if (!needsIsolatedBinding) {
            binding->buffer = buffer;
            binding->offset = bindOffset;
            /* The GL buffer's Metal backing is about to be staged in a
             * compute encoder: pin its snapshot-pool slot (P3). */
            mglNoteBufferEncoded(ptr);
            continue;
        }

        NSUInteger fallbackLength = MAX(requiredBytes, sizeof(uint32_t));
        id<MTLBuffer> isolated = [_device newBufferWithLength:fallbackLength
                                                      options:MTLResourceStorageModeShared];
        if (!isolated || !isolated.contents) {
            return false;
        }
        memset(isolated.contents, 0, fallbackLength);

        binding->buffer = isolated;
        binding->offset = 0u;
        if (buffer && availableBytes > 0) {
            binding->initialization_source = buffer;
            binding->initialization_source_offset = bindOffset;
            binding->initialization_length = MIN(availableBytes, fallbackLength);
        }

        BOOL writableResource =
            map->resource_type == SPVC_RESOURCE_TYPE_STORAGE_BUFFER ||
            map->resource_type == SPVC_RESOURCE_TYPE_ATOMIC_COUNTER;
        if (writableResource && buffer && availableBytes > 0 &&
            ![self recordStageBindingCopyBack:copyBacks
                                       atIndex:metalBindingIndex
                                     temporary:isolated
                                   destination:buffer
                             destinationBuffer:ptr
                            destinationOffset:bindOffset
                                        length:availableBytes]) {
            return false;
        }
    }

    Program *stageProgram = mglResolveProgramForStageFromState(ctx, stage);
    if (stageProgram && stageProgram->spirv[stage].needs_buffer_size_buffer) {
        uint32_t sizeConstants[31] = {0};
        for (GLuint i = 0; i < stageBufferMap.count; i++) {
            BufferMap *map = &stageBufferMap.buffers[i];
            if (!map->buf) {
                continue;
            }
            NSUInteger metalSlot = map->has_metal_binding
                ? (NSUInteger)map->metal_binding_index
                : (NSUInteger)map->buffer_base_index;
            if (metalSlot >= 31 || metalSlot == MGL_BUFFER_SIZE_BUFFER_INDEX) {
                continue;
            }
            sizeConstants[metalSlot] = (uint32_t)mglBufferMapVisibleSize(map);
        }
        bindings->size_buffer = [_device newBufferWithBytes:sizeConstants
                                                     length:sizeof(sizeConstants)
                                                    options:MTLResourceStorageModeShared];
        if (!bindings->size_buffer) {
            return false;
        }
    }

    BOOL needsInitializationBlit = NO;
    for (NSUInteger i = 0; i < kMGLMaxBufferSlots; i++) {
        if (bindings->slots[i].initialization_length > 0) {
            needsInitializationBlit = YES;
            break;
        }
    }
    if (!needsInitializationBlit) {
        return true;
    }
    if (!_renderPassManager.state->currentCommandBuffer ||
        _renderPassManager.state->currentCommandBuffer.status != MTLCommandBufferStatusNotEnqueued) {
        return false;
    }

    id<MTLBlitCommandEncoder> blit = [_renderPassManager.state->currentCommandBuffer blitCommandEncoder];
    if (!blit) {
        return false;
    }
    for (NSUInteger i = 0; i < kMGLMaxBufferSlots; i++) {
        MGLTessStageBufferBinding *binding = &bindings->slots[i];
        if (binding->initialization_length == 0) {
            continue;
        }
        [blit copyFromBuffer:binding->initialization_source
                sourceOffset:binding->initialization_source_offset
                    toBuffer:binding->buffer
           destinationOffset:0
                        size:binding->initialization_length];
    }
    [blit endEncoding];
    return true;
}

- (bool)bindPreparedTessStageBufferBindings:(const MGLTessStageBufferBindingList *)bindings
                           toComputeEncoder:(id<MTLComputeCommandEncoder>)computeCommandEncoder
{
    if (!bindings || !computeCommandEncoder) {
        return false;
    }
    for (NSUInteger i = 0; i < kMGLMaxBufferSlots; i++) {
        const MGLTessStageBufferBinding *binding = &bindings->slots[i];
        if (binding->valid) {
            [computeCommandEncoder setBuffer:binding->buffer
                                      offset:binding->offset
                                     atIndex:i];
        }
    }
    if (bindings->size_buffer) {
        [computeCommandEncoder setBuffer:bindings->size_buffer
                                  offset:0
                                 atIndex:MGL_BUFFER_SIZE_BUFFER_INDEX];
    }
    return true;
}

/* Dispatch a tessellation control shader (TCS) as a Metal compute kernel.
 * SPIRV-Cross lowers GL_TESS_CONTROL_SHADER to `kernel void` and writes
 * tessellation factors to buffer(26) and per-patch output to buffer(27).
 * Indirect params (vertexCount, instanceCount) go in buffer(29).
 *
 * For shader_image_size tests the TCS kernel only needs storage images and
 * the tess-factor / indirect-param buffers — it has no vertex input.
 */
- (void)bindPointSizeParamsToComputeEncoder:(id<MTLComputeCommandEncoder>)computeEncoder
                                    program:(Program *)program
                                      stage:(int)stage
{
    if (!computeEncoder || !program || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return;
    }
    const char *msl = program->spirv[stage].msl_str;
    if (!msl || !strstr(msl, "_mgl_point_size_params")) {
        return;
    }
    float pointSizeParams[2] = {
        ctx && MGL_STATE(ctx)->var.point_size > 0.0f ? MGL_STATE(ctx)->var.point_size : 1.0f,
        ctx && MGL_STATE(ctx)->caps.program_point_size ? 1.0f : 0.0f
    };
    [computeEncoder setBytes:pointSizeParams
                      length:sizeof(pointSizeParams)
                     atIndex:kMGLPointSizeParamBufferIndex];
}

- (id<MTLBuffer>)newTCSStageInBufferForContext:(GLMContext)drawCtx
                                       program:(Program *)tcsProgram
                                         first:(GLint)first
                                         count:(GLsizei)count
                                     indexType:(GLenum)indexType
                                       indices:(const void *)indices
                                    baseVertex:(GLint)baseVertex
                                  baseInstance:(GLuint)baseInstance
                                 patchVertices:(GLuint)patchVertices
                                    patchCount:(GLuint)patchCount
                                     outStride:(NSUInteger *)outStride
{
    if (outStride) {
        *outStride = 0u;
    }
    if (!drawCtx || !tcsProgram || count <= 0) {
        return nil;
    }

    const char *tcsMsl = tcsProgram->spirv[_TESS_CONTROL_SHADER].msl_str;
    if (!tcsMsl || !strstr(tcsMsl, "_mgl_tcs_in_buffer")) {
        return nil;
    }

    MGLTCSStageInMember members[MAX_ATTRIBS];
    memset(members, 0, sizeof(members));
    NSUInteger tcsInStride = 0u;
    NSUInteger memberCount = mglParseTCSStageInMembers(tcsMsl,
                                                       members,
                                                       MAX_ATTRIBS,
                                                       &tcsInStride);
    if (tcsInStride == 0u) {
        tcsInStride = mglComputeMSLStructSizeBySuffix(tcsMsl, "_in", 3);
    }
    if (tcsInStride == 0u) {
        NSLog(@"MGL TESS WARNING: unable to compute TCS stage_in stride for program %u",
              (unsigned)tcsProgram->name);
        return nil;
    }

    NSUInteger tcsInVertices = (NSUInteger)patchCount * (NSUInteger)MAX(patchVertices, 1u);
    if (tcsInVertices < (NSUInteger)count) {
        tcsInVertices = (NSUInteger)count;
    }
    if (tcsInVertices == 0u || tcsInStride > NSUIntegerMax / tcsInVertices) {
        return nil;
    }

    VertexArray *vao = mglRendererGetValidatedVAO(drawCtx, "tcs.stage_in");
    if (!vao) {
        return nil;
    }

    const uint8_t *indexBytes = NULL;
    NSUInteger indexOffset = (NSUInteger)(uintptr_t)indices;
    NSUInteger indexStride = 0u;
    uint32_t restartIndex = 0u;
    bool primitiveRestart = false;
    if (indexType != 0u) {
        indexStride = mglGLIndexElementSize(indexType);
        if (indexStride == 0u || indexOffset > NSUIntegerMax - ((NSUInteger)count * indexStride)) {
            return nil;
        }
        Buffer *ebo = getElementBuffer(drawCtx);
        if (!ebo || ![self processBuffer:ebo]) {
            NSLog(@"MGL TESS WARNING: TCS indexed stage_in has no readable element buffer");
            return nil;
        }
        const uint8_t *eboBytes = mglRendererReadableBufferBytes(ebo);
        NSUInteger bytesNeeded = (NSUInteger)count * indexStride;
        if (!eboBytes || indexOffset > (NSUInteger)ebo->size || ((NSUInteger)ebo->size - indexOffset) < bytesNeeded) {
            NSLog(@"MGL TESS WARNING: TCS indexed stage_in element range OOB offset=%lu needed=%lu size=%lld",
                  (unsigned long)indexOffset,
                  (unsigned long)bytesNeeded,
                  (long long)ebo->size);
            return nil;
        }
        indexBytes = eboBytes + indexOffset;
        primitiveRestart = mglPrimitiveRestartIndexForType(drawCtx, indexType, &restartIndex);
    }

    NSUInteger tcsInSize = tcsInStride * tcsInVertices;
    id<MTLBuffer> stageInBuffer = [_device newBufferWithLength:tcsInSize
                                                       options:MTLResourceStorageModeShared];
    if (!stageInBuffer) {
        return nil;
    }
    memset(stageInBuffer.contents, 0, tcsInSize);

    if (memberCount == 0u) {
        if (outStride) {
            *outStride = tcsInStride;
        }
        return stageInBuffer;
    }

    for (NSUInteger v = 0; v < tcsInVertices; v++) {
        if (v >= (NSUInteger)count) {
            continue;
        }

        int64_t vertexIndex64 = (int64_t)first + (int64_t)v;
        if (indexBytes) {
            uint32_t rawIndex = mglReadGLIndexValue(indexBytes, indexType, v);
            if (primitiveRestart && rawIndex == restartIndex) {
                continue;
            }
            vertexIndex64 = (int64_t)rawIndex + (int64_t)baseVertex;
        }
        if (vertexIndex64 < 0) {
            continue;
        }

        uint8_t *dstVertex = (uint8_t *)stageInBuffer.contents + (v * tcsInStride);
        for (NSUInteger m = 0; m < memberCount; m++) {
            const MGLTCSStageInMember *member = &members[m];
            if (member->attribute >= MAX_ATTRIBS ||
                member->offset >= tcsInStride ||
                member->size > tcsInStride - member->offset) {
                continue;
            }

            double values[4] = {0.0, 0.0, 0.0, 1.0};
            const VertexAttrib *attrib = &vao->attrib[member->attribute];
            MGLResolvedVertexAttribBinding resolved = {0};
            bool hasBinding = mglRendererResolveVertexAttribBinding(drawCtx,
                                                                    vao,
                                                                    member->attribute,
                                                                    "tcs.stage_in",
                                                                    &resolved);
            bool useCurrentValue =
                ((vao->enabled_attribs & (0x1u << member->attribute)) == 0u) &&
                !(vao->enabled_attribs == 0u && hasBinding);

            if (useCurrentValue) {
                uint8_t currentBytes[16];
                if (mglRendererBuildCurrentVertexAttribBytes(drawCtx,
                                                             member->attribute,
                                                             attrib,
                                                             currentBytes) > 0u) {
                    for (GLuint c = 0; c < MIN(attrib->size, 4u); c++) {
                        values[c] = mglDecodeVertexAttribComponent(currentBytes,
                                                                   attrib->type,
                                                                   attrib->normalized,
                                                                   c);
                    }
                }
            } else if (hasBinding) {
                Buffer *vbo = resolved.buffer;
                if (vbo && [self processBuffer:vbo]) {
                    const uint8_t *vboBytes = mglRendererReadableBufferBytes(vbo);
                    NSUInteger elementBytes = mglVertexAttribElementBytes(attrib->type, attrib->size);
                    NSUInteger stride = resolved.stride > 0u ? (NSUInteger)resolved.stride : elementBytes;
                    NSUInteger attribIndex = (NSUInteger)vertexIndex64;
                    if (resolved.divisor > 0u) {
                        attribIndex = (NSUInteger)(baseInstance / resolved.divisor);
                    }
                    if (vboBytes && elementBytes > 0u && stride > 0u &&
                        resolved.binding_offset >= 0 && resolved.relativeoffset >= 0) {
                        NSUInteger baseOffset = (NSUInteger)resolved.binding_offset + (NSUInteger)resolved.relativeoffset;
                        if (attribIndex <= (NSUIntegerMax - baseOffset) / stride) {
                            NSUInteger vertexOffset = baseOffset + attribIndex * stride;
                            if (vertexOffset <= (NSUInteger)vbo->size &&
                                ((NSUInteger)vbo->size - vertexOffset) >= elementBytes) {
                                GLboolean effectiveNormalized = attrib->normalized;
                                const uint8_t *src = vboBytes + vertexOffset;
                                for (GLuint c = 0; c < MIN(attrib->size, 4u); c++) {
                                    values[c] = mglDecodeVertexAttribComponent(src,
                                                                               attrib->type,
                                                                               effectiveNormalized,
                                                                               c);
                                }
                            }
                        }
                    }
                }
            }

            for (GLuint c = 0; c < member->components && c < 4u; c++) {
                mglWriteTCSStageInComponent(dstVertex, member, c, values[c]);
            }
        }
    }

    if (outStride) {
        *outStride = tcsInStride;
    }
    return stageInBuffer;
}

-(bool) dispatchTessControlShader:(GLMContext) glm_ctx
                          program:(Program *) tcsProgram
                            first:(GLint) first
                            count:(GLsizei) count
                        indexType:(GLenum) indexType
                          indices:(const void *) indices
                       baseVertex:(GLint) baseVertex
                     instanceCount:(GLsizei) drawInstanceCount
                     baseInstance:(GLuint) baseInstance
{
    if (!tcsProgram || !glm_ctx) {
        return false;
    }

    Shader *tcsShader = tcsProgram->shader_slots[_TESS_CONTROL_SHADER];
    if (!tcsShader || !tcsProgram->spirv[_TESS_CONTROL_SHADER].mtl_function) {
        NSLog(@"MGL TESS WARNING: TCS program %u has no compiled function", tcsProgram->name);
        return false;
    }

    /* Create compute pipeline state for TCS kernel. */
    NSError *err = nil;
    id<MTLComputePipelineState> tcsPipeline = mglGetOrCreateProgramComputePipeline(
        _device,
        tcsProgram,
        _TESS_CONTROL_SHADER,
        &err);
    if (!tcsPipeline) {
        NSLog(@"MGL TESS ERROR: failed to create TCS compute pipeline for program %u: %@",
              tcsProgram->name, err);
        return false;
    }

    /* PASS 1: Pre-resolve all Metal textures that the TCS kernel needs.
     * This must happen BEFORE we open a compute encoder, because lazy
     * Metal texture creation (bindMTLTexture:) may open its own blit
     * encoder on the command buffer, and Metal forbids two encoders
     * on the same command buffer simultaneously.  End any active render
     * encoder first for the same reason. */
    if (_renderPassManager.state->currentRenderEncoder) {
        [self endRenderEncoding];
    }

    /* Ensure a writable command buffer exists.  The GL_PATCHES path returns
     * before processGLState() (which normally creates the command buffer),
     * and prior operations (glBufferData, glEndQuery, etc.) may have
     * committed the previous command buffer. */
    if (!_renderPassManager.state->currentCommandBuffer ||
        _renderPassManager.state->currentCommandBuffer.status >= MTLCommandBufferStatusCommitted) {
        if (![self newCommandBuffer]) {
            NSLog(@"MGL TESS ERROR: failed to create command buffer for TCS dispatch");
            return false;
        }
    }

    GLuint tcsImgCount = [self getProgramBindingCount:_TESS_CONTROL_SHADER
                                                  type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE];
    for (GLuint i = 0; i < tcsImgCount; i++) {
        SpirvResource *resource = NULL;
        if (tcsProgram &&
            i < tcsProgram->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].count) {
            resource = &tcsProgram->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].list[i];
        }
        if (mglShouldSkipStageTextureResource(tcsProgram,
                                              _TESS_CONTROL_SHADER,
                                              SPVC_RESOURCE_TYPE_STORAGE_IMAGE,
                                              resource)) {
            continue;
        }
        GLuint glUnit = resource ? resource->gl_binding
                                 : [self getProgramGLBinding:_TESS_CONTROL_SHADER
                                                        type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE
                                                       index:(int)i];
        if (glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = MGL_STATE(ctx)->image_units[glUnit].tex;
        if (ptr && !ptr->mtl_data) {
            [self bindMTLTexture:ptr];
        }
    }

    MGLStageBindingCopyBackList stageCopyBacks = {0};
    MGLTessStageBufferBindingList stageBufferBindings = {0};
    if (![self prepareTessStageBufferBindings:&stageBufferBindings
                                         stage:_TESS_CONTROL_SHADER
                                     copyBacks:&stageCopyBacks]) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    id<MTLComputeCommandEncoder> computeEncoder = [_renderPassManager.state->currentCommandBuffer computeCommandEncoder];
    if (!computeEncoder) {
        NSLog(@"MGL TESS ERROR: failed to create compute encoder for TCS dispatch");
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    [computeEncoder setComputePipelineState:tcsPipeline];

    /* PASS 2: Bind storage images for TCS stage. */
    for (GLuint i = 0; i < tcsImgCount; i++) {
        SpirvResource *resource = NULL;
        if (tcsProgram &&
            i < tcsProgram->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].count) {
            resource = &tcsProgram->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].list[i];
        }
        if (mglShouldSkipStageTextureResource(tcsProgram,
                                              _TESS_CONTROL_SHADER,
                                              SPVC_RESOURCE_TYPE_STORAGE_IMAGE,
                                              resource)) {
            continue;
        }
        GLuint metalSlot = resource ? mglMetalResourceSlot(resource)
                                    : [self getProgramBinding:_TESS_CONTROL_SHADER
                                                        type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE
                                                       index:(int)i];
        GLuint glUnit = resource ? resource->gl_binding
                                 : [self getProgramGLBinding:_TESS_CONTROL_SHADER
                                                        type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE
                                                       index:(int)i];
        if (metalSlot >= TEXTURE_UNITS || glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = MGL_STATE(ctx)->image_units[glUnit].tex;
        id<MTLTexture> texture = nil;
        if (ptr) {
            texture = (__bridge id<MTLTexture>)(ptr->mtl_data);
            GLuint imgLevel = MGL_STATE(ctx)->image_units[glUnit].level;
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
        [computeEncoder setTexture:texture atIndex:metalSlot];
    }

    /* Also bind sampled (read-only) images for TCS stage. */
    GLuint tcsSampledCount = [self getProgramBindingCount:_TESS_CONTROL_SHADER
                                                     type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];
    for (GLuint i = 0; i < tcsSampledCount; i++) {
        SpirvResource *resource = NULL;
        if (tcsProgram &&
            i < tcsProgram->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE].count) {
            resource = &tcsProgram->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE].list[i];
        }
        if (mglShouldSkipStageTextureResource(tcsProgram,
                                              _TESS_CONTROL_SHADER,
                                              SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
                                              resource)) {
            continue;
        }
        GLuint metalSlot = resource ? mglMetalResourceSlot(resource)
                                    : [self getProgramBinding:_TESS_CONTROL_SHADER
                                                        type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                       index:(int)i];
        GLuint glUnit = resource ? resource->gl_binding
                                 : [self getProgramGLBinding:_TESS_CONTROL_SHADER
                                                        type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                       index:(int)i];
        if (metalSlot >= TEXTURE_UNITS || glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = MGL_STATE(ctx)->active_textures[glUnit];
        if (ptr && !ptr->mtl_data) {
            [self bindMTLTexture:ptr];
        }
        id<MTLTexture> texture = ptr ? (__bridge id<MTLTexture>)(ptr->mtl_data) : nil;
        [computeEncoder setTexture:texture atIndex:metalSlot];
        if (resource && resource->msl_has_combined_sampler) {
            id<MTLSamplerState> sampler = nil;
            if (MGL_STATE(ctx)->texture_samplers[glUnit]) {
                Sampler *glSampler = MGL_STATE(ctx)->texture_samplers[glUnit];
                if (glSampler->dirty_bits && glSampler->mtl_data) {
                    mglSafeReleaseMetalObj((void **)&glSampler->mtl_data);
                }
                if (!glSampler->mtl_data && ptr) {
                    glSampler->mtl_data = (void *)CFBridgingRetain(
                        [self createMTLSamplerForTexParam:&glSampler->params
                                                  target:ptr->target]);
                    glSampler->dirty_bits = 0;
                }
                sampler = (__bridge id<MTLSamplerState>)(glSampler->mtl_data);
            } else if (ptr && ptr->params.mtl_data) {
                sampler = (__bridge id<MTLSamplerState>)(ptr->params.mtl_data);
            }
            if (!sampler) {
                sampler = [_device newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]];
            }
            if (sampler) {
                [computeEncoder setSamplerState:sampler
                                        atIndex:mglMetalCombinedSamplerSlot(resource)];
            }
        }
    }

    /* Bind stage buffers (UBO, SSBO, atomic counters) for TCS. */
    if (![self bindPreparedTessStageBufferBindings:&stageBufferBindings
                                  toComputeEncoder:computeEncoder]) {
        [computeEncoder endEncoding];
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    [self bindPointSizeParamsToComputeEncoder:computeEncoder
                                      program:tcsProgram
                                        stage:_TESS_CONTROL_SHADER];

    /* Create indirect params buffer (buffer 29).
     * spvIndirectParams[0] = vertexCount, [1] = instanceCount. */
    GLuint patchVertices = MAX(1u, (GLuint)MGL_STATE(ctx)->var.patch_vertices);
    GLuint vertexCount = (GLuint)count;
    GLuint instanceCount = (drawInstanceCount > 0) ? (GLuint)drawInstanceCount : 1u;

    /* Create TCS per-vertex output buffer (buffer 28 = spvOut).
     * TCS writes: spvOut[gl_PrimitiveID * outputVertices + invocationID]
     * where outputVertices = tess_control_output_vertices (layout(vertices=N) out).
     * Compute the per-vertex stride from the TCS stage output resources. */
    GLuint tcsOutVertices = tcsProgram->tess_control_output_vertices;
    if (tcsOutVertices == 0) tcsOutVertices = patchVertices;

    /* Compute the per-vertex stride by parsing the MSL output wrapper struct.
     * This includes built-in outputs (gl_Position, gl_PointSize, ...) and
     * Metal alignment padding, which the SPIRV-Cross resource list omits. */
    NSUInteger tcsOutStride = 0;
    const char *tcsMsl = tcsProgram->spirv[_TESS_CONTROL_SHADER].msl_str;
    if (tcsMsl) {
        tcsOutStride = mglComputeMSLOutputStructSize(tcsMsl);
    }
    /* Fallback: sum user-defined outputs from the resource list. */
    if (tcsOutStride == 0 && tcsProgram) {
        SpirvResourceList *outs =
            &tcsProgram->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT];
        for (GLuint i = 0; outs->list && i < outs->count; i++) {
            GLenum gt = outs->list[i].gl_type;
            GLuint comps = 1, bytesPer = 4;
            if (gt == GL_FLOAT_VEC4 || gt == GL_INT_VEC4 || gt == GL_UNSIGNED_INT_VEC4 ||
                gt == GL_BOOL_VEC4) { comps = 4; }
            else if (gt == GL_FLOAT_VEC3 || gt == GL_INT_VEC3 || gt == GL_UNSIGNED_INT_VEC3 ||
                     gt == GL_BOOL_VEC3) { comps = 3; }
            else if (gt == GL_FLOAT_VEC2 || gt == GL_INT_VEC2 || gt == GL_UNSIGNED_INT_VEC2 ||
                     gt == GL_BOOL_VEC2) { comps = 2; }
            else if (gt == GL_FLOAT || gt == GL_INT || gt == GL_UNSIGNED_INT || gt == GL_BOOL) { comps = 1; }
            else if (gt == GL_DOUBLE_VEC4 || gt == GL_DOUBLE_VEC3 || gt == GL_DOUBLE_VEC2) {
                comps = (gt == GL_DOUBLE_VEC4) ? 4 : (gt == GL_DOUBLE_VEC3 ? 3 : 2);
                bytesPer = 8;
            }
            else { comps = 4; }
            tcsOutStride += comps * bytesPer;
        }
    }
    if (tcsOutStride == 0) tcsOutStride = 64;  /* fallback: 4 x float4 */
    _tessellation.tcsOutputStride = tcsOutStride;
    _tessellation.tcsOutVertices = tcsOutVertices;

    GLuint patchCountTC = vertexCount / patchVertices;
    if (patchCountTC == 0u) patchCountTC = 1u;
    NSUInteger tcsOutSize = (NSUInteger)patchCountTC * tcsOutVertices * tcsOutStride;
    _tessellation.tcsOutputBuffer = [_device newBufferWithLength:tcsOutSize
                                            options:MTLResourceStorageModeShared];
    if (!_tessellation.tcsOutputBuffer || !_tessellation.tcsOutputBuffer.contents) {
        [computeEncoder endEncoding];
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    memset(_tessellation.tcsOutputBuffer.contents, 0, tcsOutSize);
    [computeEncoder setBuffer:_tessellation.tcsOutputBuffer offset:0 atIndex:28];

    /* Create TCS per-patch output buffer (buffer 27 = spvPatchOut).
     * TCS writes: spvPatchOut[gl_PrimitiveID].
     * The per-patch struct size is harder to compute generically; use a
     * generous estimate based on the patch output resources. */
    NSUInteger tcsPatchStride = 0;
    if (tcsProgram) {
        /* Per-patch outputs share SPVC_RESOURCE_TYPE_STAGE_OUTPUT with
         * per-vertex outputs; SpvDecorationPatch is reflected as is_per_patch. */
        SpirvResourceList *outs =
            &tcsProgram->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT];
        for (GLuint i = 0; outs->list && i < outs->count; i++) {
            if (!outs->list[i].is_per_patch) continue;
            GLenum gt = outs->list[i].gl_type;
            GLuint comps = 1, bytesPer = 4;
            if (gt == GL_FLOAT_VEC4 || gt == GL_INT_VEC4 || gt == GL_UNSIGNED_INT_VEC4) { comps = 4; }
            else if (gt == GL_FLOAT_VEC3 || gt == GL_INT_VEC3 || gt == GL_UNSIGNED_INT_VEC3) { comps = 3; }
            else if (gt == GL_FLOAT_VEC2 || gt == GL_INT_VEC2 || gt == GL_UNSIGNED_INT_VEC2) { comps = 2; }
            else if (gt == GL_FLOAT || gt == GL_INT || gt == GL_UNSIGNED_INT) { comps = 1; }
            else { comps = 4; }
            tcsPatchStride += comps * bytesPer;
        }
    }
    if (tcsPatchStride == 0) tcsPatchStride = 16;  /* fallback: 1 x float4 */
    NSUInteger tcsPatchSize = (NSUInteger)patchCountTC * tcsPatchStride;
    _tessellation.tcsPatchOutBuffer = [_device newBufferWithLength:tcsPatchSize
                                              options:MTLResourceStorageModeShared];
    if (!_tessellation.tcsPatchOutBuffer || !_tessellation.tcsPatchOutBuffer.contents) {
        [computeEncoder endEncoding];
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    memset(_tessellation.tcsPatchOutBuffer.contents, 0, tcsPatchSize);
    [computeEncoder setBuffer:_tessellation.tcsPatchOutBuffer offset:0 atIndex:27];

    GLuint indirectParams[2] = { patchVertices, instanceCount };
    id<MTLBuffer> indirectBuf = [_device newBufferWithBytes:indirectParams
                                                     length:sizeof(indirectParams)
                                                    options:MTLResourceStorageModeShared];
    if (!indirectBuf) {
        [computeEncoder endEncoding];
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    [computeEncoder setBuffer:indirectBuf offset:0 atIndex:29];

    /* Create tessellation factor buffer (buffer 26).
     * MTLQuadTessellationFactorsHalf = 4 edge + 2 inner half-floats = 12 bytes/patch. */
    GLuint patchCount = vertexCount / patchVertices;
    if (patchCount == 0u) patchCount = 1u;
    NSUInteger tessFactorSize = (NSUInteger)patchCount * 12u;
    id<MTLBuffer> tessFactorBuf = [_device newBufferWithLength:tessFactorSize
                                                       options:MTLResourceStorageModeShared];
    if (!tessFactorBuf || !tessFactorBuf.contents) {
        [computeEncoder endEncoding];
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    memset(tessFactorBuf.contents, 0, tessFactorSize);
    [computeEncoder setBuffer:tessFactorBuf offset:0 atIndex:26];

    if (tcsMsl && strstr(tcsMsl, "_mgl_tcs_in_buffer")) {
        NSUInteger tcsInStride = 0u;
        id<MTLBuffer> tcsStageInBuffer =
            [self newTCSStageInBufferForContext:glm_ctx
                                        program:tcsProgram
                                          first:first
                                          count:count
                                      indexType:indexType
                                        indices:indices
                                     baseVertex:baseVertex
                                   baseInstance:baseInstance
                                  patchVertices:patchVertices
                                     patchCount:patchCount
                                      outStride:&tcsInStride];
        if (!tcsStageInBuffer) {
            NSLog(@"MGL TESS WARNING: failed to pack TCS stage_in buffer for program %u",
                  tcsProgram ? (unsigned)tcsProgram->name : 0u);
            [computeEncoder endEncoding];
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }
        [computeEncoder setBuffer:tcsStageInBuffer
                            offset:0
                           atIndex:kMGLTCSStageInReplBufferIndex];

        /* SPIRV-Cross emits `threadgroup <in_type>* gl_in [[threadgroup(0)]]`
         * for TCS with per-vertex inputs; Metal requires the host to size it
         * via setThreadgroupMemoryLength or the whole command buffer faults
         * ("missing Threadgroup Memory binding") and every later dispatch's
         * writes are dropped.  gl_in is indexed up to
         * max(patchVertices, tcsOutVertices) with the stage_in element
         * layout, so tcsInStride is the element size. */
        if (strstr(tcsMsl, "[[threadgroup(0)]]")) {
            NSUInteger tgElems = MAX((NSUInteger)patchVertices,
                                     (NSUInteger)tcsOutVertices);
            NSUInteger tgStride = tcsInStride ? tcsInStride : 64u;
            NSUInteger tgLength = (tgElems * tgStride + 15u) & ~(NSUInteger)15u;
            [computeEncoder setThreadgroupMemoryLength:tgLength atIndex:0];
        }
    }

    /* Dispatch: one threadgroup per patch, tcsOutVertices threads per threadgroup (one thread per TCS output vertex = gl_InvocationID). */
    MTLSize threadgroups = MTLSizeMake(patchCount, 1, 1);
    MTLSize threadsPerTG = MTLSizeMake(tcsOutVertices, 1, 1);
    [computeEncoder dispatchThreadgroups:threadgroups
                     threadsPerThreadgroup:threadsPerTG];

    [computeEncoder endEncoding];

    if (![self flushStageBindingCopyBacks:&stageCopyBacks
                     requireCPUVisibility:YES]) {
        NSLog(@"MGL TESS ERROR: failed to synchronize TCS buffer writes");
        return false;
    }

    /* Save tess factor buffer for TES drawPatches path. */
    _tessellation.tessFactorBuffer = tessFactorBuf;

    return true;
}

static NSUInteger mglTESXFBFieldByteSize(GLenum glType)
{
    switch (glType) {
        case GL_FLOAT:
        case GL_INT:
        case GL_UNSIGNED_INT:
            return 4u;
        case GL_FLOAT_VEC2:
        case GL_INT_VEC2:
        case GL_UNSIGNED_INT_VEC2:
            return 8u;
        case GL_FLOAT_VEC3:
        case GL_INT_VEC3:
        case GL_UNSIGNED_INT_VEC3:
            return 12u;
        case GL_FLOAT_VEC4:
        case GL_INT_VEC4:
        case GL_UNSIGNED_INT_VEC4:
            return 16u;
        default:
            return 0u;
    }
}

/* Keep this layout calculation in lockstep with the packed writes injected by
 * mglFixMSLTesAsComputeKernel.  A zero result means the renderer cannot prove
 * the write stride and must not copy temporary capture data into the GL store. */
static NSUInteger mglTESXFBVertexStride(const Program *program)
{
    if (!program || program->transform_feedback_varying_count <= 0) {
        return 0u;
    }

    const SpirvResourceList *outputs =
        &program->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT];
    NSUInteger stride = 0u;
    for (GLsizei varying = 0;
         varying < program->transform_feedback_varying_count;
         varying++) {
        const char *name = program->transform_feedback_varying_names[varying];
        const SpirvResource *output = NULL;
        for (GLuint i = 0; name && outputs->list && i < outputs->count; i++) {
            if (outputs->list[i].name && strcmp(outputs->list[i].name, name) == 0) {
                output = &outputs->list[i];
                break;
            }
        }

        NSUInteger fieldBytes = output ? mglTESXFBFieldByteSize(output->gl_type) : 0u;
        if (fieldBytes == 0u || stride > NSUIntegerMax - fieldBytes) {
            return 0u;
        }
        stride += fieldBytes;
    }
    return stride;
}

static bool mglCheckedNSUIntegerProduct(NSUInteger a,
                                        NSUInteger b,
                                        NSUInteger *result)
{
    if (!result || (a != 0u && b > NSUIntegerMax / a)) {
        return false;
    }
    *result = a * b;
    return true;
}

/* Dispatch a TES (Tessellation Evaluation Shader) when there is no TCS and
 * GL_RASTERIZER_DISCARD is active.  SPIRV-Cross lowers the TES to a Metal
 * post-tessellation vertex function (`[[patch(quad, 0)]] vertex ...`), but
 * macOS 26.5 SDK removed postTessellationVertexFunction / isTessellationEnabled
 * from MTLRenderPipelineDescriptor.  We therefore rewrite the TES MSL to a
 * plain compute kernel (mglFixMSLTesAsComputeKernel in program.c) and dispatch
 * it with a compute pipeline, exactly like TCS.
 *
 * The TES kernel uses gl_PrimitiveID (mapped to threadgroup_position_in_grid)
 * as the patch index.  We dispatch one threadgroup per patch with 1 thread
 * per threadgroup, so each invocation handles one patch. */
-(bool) dispatchTessEvaluationShader:(GLMContext) glm_ctx
                            program:(Program *) tesProgram
                              first:(GLint) first
                              count:(GLsizei) count
{
    if (!tesProgram || !glm_ctx) {
        return false;
    }

    Shader *tesShader = tesProgram->shader_slots[_TESS_EVALUATION_SHADER];
    if (!tesShader || !tesProgram->spirv[_TESS_EVALUATION_SHADER].mtl_function) {
        NSLog(@"MGL TESS WARNING: TES program %u has no compiled function", tesProgram->name);
        return false;
    }

    /* Create compute pipeline state for TES kernel. */
    NSError *err = nil;
    id<MTLComputePipelineState> tesPipeline = mglGetOrCreateProgramComputePipeline(
        _device,
        tesProgram,
        _TESS_EVALUATION_SHADER,
        &err);
    if (!tesPipeline) {
        NSLog(@"MGL TESS ERROR: failed to create TES compute pipeline for program %u: %@",
              tesProgram->name, err);
        return false;
    }

    /* PASS 1: Pre-resolve all Metal textures that the TES kernel needs.
     * Must happen before opening any encoder (same reason as TCS). */
    if (_renderPassManager.state->currentRenderEncoder) {
        [self endRenderEncoding];
    }

    /* Ensure a writable command buffer exists (same reason as TCS). */
    if (!_renderPassManager.state->currentCommandBuffer ||
        _renderPassManager.state->currentCommandBuffer.status >= MTLCommandBufferStatusCommitted) {
        if (![self newCommandBuffer]) {
            NSLog(@"MGL TESS ERROR: failed to create command buffer for TES dispatch");
            return false;
        }
    }

    GLuint tesImgCount = [self getProgramBindingCount:_TESS_EVALUATION_SHADER
                                                  type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE];
    for (GLuint i = 0; i < tesImgCount; i++) {
        SpirvResource *resource = NULL;
        if (tesProgram &&
            i < tesProgram->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].count) {
            resource = &tesProgram->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].list[i];
        }
        if (mglShouldSkipStageTextureResource(tesProgram,
                                              _TESS_EVALUATION_SHADER,
                                              SPVC_RESOURCE_TYPE_STORAGE_IMAGE,
                                              resource)) {
            continue;
        }
        GLuint glUnit = resource ? resource->gl_binding
                                 : [self getProgramGLBinding:_TESS_EVALUATION_SHADER
                                                        type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE
                                                       index:(int)i];
        if (glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = MGL_STATE(ctx)->image_units[glUnit].tex;
        if (ptr && !ptr->mtl_data) {
            [self bindMTLTexture:ptr];
        }
    }

    MGLStageBindingCopyBackList stageCopyBacks = {0};
    MGLTessStageBufferBindingList stageBufferBindings = {0};
    if (![self prepareTessStageBufferBindings:&stageBufferBindings
                                         stage:_TESS_EVALUATION_SHADER
                                     copyBacks:&stageCopyBacks]) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    id<MTLComputeCommandEncoder> computeEncoder = [_renderPassManager.state->currentCommandBuffer computeCommandEncoder];
    if (!computeEncoder) {
        NSLog(@"MGL TESS ERROR: failed to create compute encoder for TES dispatch");
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    [computeEncoder setComputePipelineState:tesPipeline];

    /* PASS 2: Bind storage images for TES stage. */
    for (GLuint i = 0; i < tesImgCount; i++) {
        SpirvResource *resource = NULL;
        if (tesProgram &&
            i < tesProgram->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].count) {
            resource = &tesProgram->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].list[i];
        }
        if (mglShouldSkipStageTextureResource(tesProgram,
                                              _TESS_EVALUATION_SHADER,
                                              SPVC_RESOURCE_TYPE_STORAGE_IMAGE,
                                              resource)) {
            continue;
        }
        GLuint metalSlot = resource ? mglMetalResourceSlot(resource)
                                    : [self getProgramBinding:_TESS_EVALUATION_SHADER
                                                        type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE
                                                       index:(int)i];
        GLuint glUnit = resource ? resource->gl_binding
                                 : [self getProgramGLBinding:_TESS_EVALUATION_SHADER
                                                        type:SPVC_RESOURCE_TYPE_STORAGE_IMAGE
                                                       index:(int)i];
        if (metalSlot >= TEXTURE_UNITS || glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = MGL_STATE(ctx)->image_units[glUnit].tex;
        id<MTLTexture> texture = nil;
        if (ptr) {
            texture = (__bridge id<MTLTexture>)(ptr->mtl_data);
            GLuint imgLevel = MGL_STATE(ctx)->image_units[glUnit].level;
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
        [computeEncoder setTexture:texture atIndex:metalSlot];
    }

    /* Also bind sampled (read-only) images for TES stage. */
    GLuint tesSampledCount = [self getProgramBindingCount:_TESS_EVALUATION_SHADER
                                                     type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];
    for (GLuint i = 0; i < tesSampledCount; i++) {
        SpirvResource *resource = NULL;
        if (tesProgram &&
            i < tesProgram->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE].count) {
            resource = &tesProgram->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE].list[i];
        }
        if (mglShouldSkipStageTextureResource(tesProgram,
                                              _TESS_EVALUATION_SHADER,
                                              SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
                                              resource)) {
            continue;
        }
        GLuint metalSlot = resource ? mglMetalResourceSlot(resource)
                                    : [self getProgramBinding:_TESS_EVALUATION_SHADER
                                                        type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                       index:(int)i];
        GLuint glUnit = resource ? resource->gl_binding
                                 : [self getProgramGLBinding:_TESS_EVALUATION_SHADER
                                                        type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                       index:(int)i];
        if (metalSlot >= TEXTURE_UNITS || glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = MGL_STATE(ctx)->active_textures[glUnit];
        if (ptr && !ptr->mtl_data) {
            [self bindMTLTexture:ptr];
        }
        id<MTLTexture> texture = ptr ? (__bridge id<MTLTexture>)(ptr->mtl_data) : nil;
        [computeEncoder setTexture:texture atIndex:metalSlot];
        if (resource && resource->msl_has_combined_sampler) {
            id<MTLSamplerState> sampler = nil;
            if (MGL_STATE(ctx)->texture_samplers[glUnit]) {
                Sampler *glSampler = MGL_STATE(ctx)->texture_samplers[glUnit];
                if (glSampler->dirty_bits && glSampler->mtl_data) {
                    mglSafeReleaseMetalObj((void **)&glSampler->mtl_data);
                }
                if (!glSampler->mtl_data && ptr) {
                    glSampler->mtl_data = (void *)CFBridgingRetain(
                        [self createMTLSamplerForTexParam:&glSampler->params
                                                  target:ptr->target]);
                    glSampler->dirty_bits = 0;
                }
                sampler = (__bridge id<MTLSamplerState>)(glSampler->mtl_data);
            } else if (ptr && ptr->params.mtl_data) {
                sampler = (__bridge id<MTLSamplerState>)(ptr->params.mtl_data);
            }
            if (!sampler) {
                sampler = [_device newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]];
            }
            if (sampler) {
                [computeEncoder setSamplerState:sampler
                                        atIndex:mglMetalCombinedSamplerSlot(resource)];
            }
        }
    }

    /* Bind stage buffers (UBO, SSBO, atomic counters) for TES. */
    if (![self bindPreparedTessStageBufferBindings:&stageBufferBindings
                                  toComputeEncoder:computeEncoder]) {
        [computeEncoder endEncoding];
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    [self bindPointSizeParamsToComputeEncoder:computeEncoder
                                      program:tesProgram
                                        stage:_TESS_EVALUATION_SHADER];

    /* Dispatch: one threadgroup per patch, 1 thread per threadgroup.
     * gl_PrimitiveID → threadgroup_position_in_grid gives the patch index.
     * TessCoord → thread_position_in_threadgroup is 0 (1 thread per TG). */
    GLuint patchVertices = MAX(1u, (GLuint)MGL_STATE(ctx)->var.patch_vertices);
    GLuint vertexCount = (GLuint)count;
    GLuint patchCount = vertexCount / patchVertices;
    if (patchCount == 0u) patchCount = 1u;

    /* Bind patch info to buffer(28): {patch_vertices_in, tcs_out_vertices}.
     * _mgl_patch_info.x = patch vertices (gl_in.size() replacement)
     * _mgl_patch_info.y = TCS output vertices per patch (for per-patch gl_in indexing) */
    {
        GLuint patchInfo[2] = { patchVertices, _tessellation.tcsOutVertices };
        if (patchInfo[1] == 0) patchInfo[1] = patchVertices;
        [computeEncoder setBytes:patchInfo length:sizeof(patchInfo) atIndex:28];
    }

    /* Bind TCS output buffer to buffer(30) for TES gl_in.
     * TCS writes per-vertex output to spvOut (buffer 28 in TCS).  TES reads
     * gl_in[...] from buffer(30).  The data layout is: TCS writes
     * spvOut[patchID * outputVertices + invocationID], so TES gl_in should
     * point to the same buffer.  The MSL rewriter changed TES's [[stage_in]]
     * to "device <type> *gl_in [[buffer(30)]]". */
    if (_tessellation.tcsOutputBuffer) {
        [computeEncoder setBuffer:_tessellation.tcsOutputBuffer offset:0 atIndex:30];
    }

    /* Bind TCS patch output buffer to buffer(27) for TES patchIn.
     * TCS writes per-patch output to spvPatchOut (buffer 27 in TCS).  TES
     * reads patchIn[...] from buffer(27).  Note: buffer 27 is reused for both
     * TCS spvPatchOut and TES patchIn, which is correct since the data flows
     * TCS → TES. */
    if (_tessellation.tcsPatchOutBuffer) {
        [computeEncoder setBuffer:_tessellation.tcsPatchOutBuffer offset:0 atIndex:27];
    }

    /* Compute vertsPerPatch from tessellation factors.
     * We dispatch vertsPerPatch threads per threadgroup so each thread
     * writes one XFB entry.  The vertex count formula matches what the
     * CTS counter program expects (primitive count * vertices-per-primitive). */
    GLuint vertsPerPatch = 1;
    if (_tessellation.tessFactorBuffer) {
        const struct {
            uint16_t edge[4];
            uint16_t inside[2];
        } __attribute__((packed)) *tf = (const void *)_tessellation.tessFactorBuffer.contents;
        GLenum genMode = tesProgram ? tesProgram->tess_gen_mode : GL_TRIANGLES;
        GLboolean pointMode = tesProgram ? tesProgram->tess_gen_point_mode : GL_FALSE;
        if (patchCount > 0) {
            float edge0 = *(const __fp16 *)&tf[0].edge[0];
            float inside0 = *(const __fp16 *)&tf[0].inside[0];
            float inside1 = *(const __fp16 *)&tf[0].inside[1];
            if (edge0 < 1.0f) edge0 = 1.0f;
            if (inside0 < 1.0f) inside0 = 1.0f;
            if (inside1 < 1.0f) inside1 = 1.0f;
            GLuint primPerPatch = 1;
            if (genMode == GL_QUADS) {
                primPerPatch = 2u * (GLuint)ceilf(inside0) * (GLuint)ceilf(inside1);
            } else if (genMode == GL_TRIANGLES) {
                primPerPatch = (GLuint)ceilf(inside0) * (GLuint)ceilf(inside0);
            } else { /* GL_ISOLINES */
                primPerPatch = (GLuint)ceilf(edge0);
            }
            if (primPerPatch == 0u) primPerPatch = 1u;
            if (pointMode) {
                vertsPerPatch = primPerPatch;
            } else if (genMode == GL_ISOLINES) {
                vertsPerPatch = primPerPatch * 2u;
            } else {
                vertsPerPatch = primPerPatch * 3u;
            }
        }
    }
    if (vertsPerPatch == 0) vertsPerPatch = 1;

    /* Bind XFB output buffer to buffer(29) for _mgl_xfb_out. Metal buffer
     * arguments cannot express a subrange, so a direct binding is safe only
     * when every injected write fits in both the requested GL range and the
     * current logical store. On overflow, capture into a full-size temporary
     * buffer and copy back only the prefix containing complete primitives. */
    TransformFeedback *xfbState = MGL_STATE(glm_ctx)->transform_feedback;
    const bool xfbCaptureActive =
        tesProgram->transform_feedback_varying_count > 0 &&
        tesProgram->transform_feedback_buffer_mode == GL_INTERLEAVED_ATTRIBS &&
        xfbState &&
        xfbState->active &&
        !xfbState->paused &&
        tesProgram->spirv[_TESS_EVALUATION_SHADER].msl_str &&
        strstr(tesProgram->spirv[_TESS_EVALUATION_SHADER].msl_str, "_mgl_xfb_out");
    id<MTLBuffer> xfbTemporary = nil;
    id<MTLBuffer> xfbCopyDestination = nil;
    Buffer *xfbDestination = NULL;
    NSUInteger xfbCopyDestinationOffset = 0u;
    NSUInteger xfbCopyBytes = 0u;
    NSUInteger xfbPrimitiveCapacity = 0u;
    NSUInteger xfbWrittenBytes = 0u;

    if (xfbCaptureActive) {
        BufferBaseTarget *xfbSlot =
            &MGL_STATE(glm_ctx)->buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[0];
        NSUInteger xfbStride = mglTESXFBVertexStride(tesProgram);
        NSUInteger conservativeStride =
            (NSUInteger)tesProgram->transform_feedback_varying_count * 16u;
        NSUInteger allocationStride = xfbStride ? xfbStride : conservativeStride;
        NSUInteger captureVertices = 0u;
        NSUInteger requiredBytes = 0u;
        NSUInteger xfbSessionOffset = 0u;
        bool sessionOffsetOK =
            xfbState->buffer_write_offsets[0] <= (GLuint64)NSUIntegerMax;
        if (sessionOffsetOK) {
            xfbSessionOffset = (NSUInteger)xfbState->buffer_write_offsets[0];
        }
        bool sizeOK =
            allocationStride > 0u &&
            mglCheckedNSUIntegerProduct((NSUInteger)patchCount,
                                        (NSUInteger)vertsPerPatch,
                                        &captureVertices) &&
            mglCheckedNSUIntegerProduct(captureVertices,
                                        allocationStride,
                                        &requiredBytes) &&
            requiredBytes > 0u;

        id<MTLBuffer> xfbMTL = nil;
        NSUInteger visibleBytes = 0u;
        NSUInteger remainingVisibleBytes = 0u;
        NSUInteger destinationOffset = 0u;
        bool destinationOffsetOK = false;
        if (xfbSlot->buf) {
            if (!xfbSlot->buf->data.mtl_data) {
                [self bindMTLBuffer:xfbSlot->buf];
            }
            xfbMTL = (__bridge id<MTLBuffer>)(xfbSlot->buf->data.mtl_data);
            if (xfbMTL) {
                BufferMap xfbMap = {0};
                xfbMap.buf = xfbSlot->buf;
                xfbMap.offset = xfbSlot->offset;
                xfbMap.size = xfbSlot->size;
                visibleBytes = mglBufferMapVisibleBackingBytes(
                    &xfbMap, (size_t)xfbMTL.length);
                if (sessionOffsetOK && xfbSessionOffset <= visibleBytes &&
                    xfbSlot->offset >= 0 &&
                    (NSUInteger)xfbSlot->offset <= NSUIntegerMax - xfbSessionOffset) {
                    remainingVisibleBytes = visibleBytes - xfbSessionOffset;
                    destinationOffset = (NSUInteger)xfbSlot->offset + xfbSessionOffset;
                    destinationOffsetOK = true;
                }
            }
        }

        if (!sizeOK) {
            NSLog(@"MGL TESS XFB: capture size overflow for program %u",
                  (unsigned)tesProgram->name);
            [computeEncoder endEncoding];
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }

        GLuint verticesPerPrimitive =
            tesProgram->tess_gen_point_mode ? 1u :
            (tesProgram->tess_gen_mode == GL_ISOLINES ? 2u : 3u);
        NSUInteger primitiveBytes = 0u;
        bool primitiveLayoutOK =
            xfbStride != 0u &&
            mglCheckedNSUIntegerProduct(xfbStride,
                                        (NSUInteger)verticesPerPrimitive,
                                        &primitiveBytes) &&
            primitiveBytes > 0u;

        if (primitiveLayoutOK && xfbMTL && destinationOffsetOK &&
            requiredBytes <= remainingVisibleBytes) {
            [computeEncoder setBuffer:xfbMTL
                               offset:destinationOffset
                              atIndex:kMGLBufferSlot_IndirectParams];
            xfbSlot->buf->ever_written = GL_TRUE;
            xfbPrimitiveCapacity = captureVertices / verticesPerPrimitive;
            xfbWrittenBytes = xfbPrimitiveCapacity * primitiveBytes;
        } else {
            xfbTemporary = [_device newBufferWithLength:requiredBytes
                                                options:MTLResourceStorageModeShared];
            if (!xfbTemporary) {
                NSLog(@"MGL TESS XFB: failed to allocate %lu-byte overflow buffer",
                      (unsigned long)requiredBytes);
                [computeEncoder endEncoding];
                [self clearStageBindingCopyBacks:&stageCopyBacks];
                return false;
            }
            memset(xfbTemporary.contents, 0, requiredBytes);
            [computeEncoder setBuffer:xfbTemporary
                               offset:0
                              atIndex:kMGLBufferSlot_IndirectParams];

            /* Unknown layouts stay in the temporary buffer. This is an honest
             * no-capture fallback; copying an unproven stride could overwrite
             * bytes outside a complete transform-feedback primitive. */
            if (primitiveLayoutOK && xfbMTL && destinationOffsetOK &&
                remainingVisibleBytes >= primitiveBytes) {
                xfbPrimitiveCapacity = MIN(captureVertices / verticesPerPrimitive,
                                           remainingVisibleBytes / primitiveBytes);
                xfbCopyBytes = xfbPrimitiveCapacity * primitiveBytes;
                xfbWrittenBytes = xfbCopyBytes;
                xfbCopyDestination = xfbMTL;
                xfbCopyDestinationOffset = destinationOffset;
                xfbDestination = xfbSlot->buf;
            }
        }
    }

    MTLSize threadgroups = MTLSizeMake(patchCount, 1, 1);
    MTLSize threadsPerTG = MTLSizeMake(vertsPerPatch, 1, 1);
    [computeEncoder dispatchThreadgroups:threadgroups
                     threadsPerThreadgroup:threadsPerTG];

    [computeEncoder endEncoding];
    /* Without this, a TES dispatch with no copy-backs stays in the current
     * command buffer and flushCommandBufferLocked's empty-CB skip drops it:
     * glFinish then never executes the TES writes (SSBO stores vanish). */
    _currentCBHasWork = YES;

    if (![self flushStageBindingCopyBacks:&stageCopyBacks
                     requireCPUVisibility:NO]) {
        NSLog(@"MGL TESS ERROR: failed to copy isolated TES writable buffer prefixes");
        return false;
    }

    if (xfbCopyBytes > 0u) {
        id<MTLBlitCommandEncoder> xfbBlit = [_renderPassManager.state->currentCommandBuffer blitCommandEncoder];
        if (!xfbBlit) {
            NSLog(@"MGL TESS XFB: failed to create bounded copy encoder");
            return false;
        }
        [xfbBlit copyFromBuffer:xfbTemporary
                   sourceOffset:0
                       toBuffer:xfbCopyDestination
              destinationOffset:xfbCopyDestinationOffset
                           size:xfbCopyBytes];
        [xfbBlit endEncoding];
        if (xfbDestination) {
            xfbDestination->ever_written = GL_TRUE;
        }
    }

    if (xfbCaptureActive && xfbWrittenBytes > 0u) {
        GLuint64 currentOffset = xfbState->buffer_write_offsets[0];
        if ((GLuint64)xfbWrittenBytes > UINT64_MAX - currentOffset) {
            xfbState->buffer_write_offsets[0] = UINT64_MAX;
        } else {
            xfbState->buffer_write_offsets[0] =
                currentOffset + (GLuint64)xfbWrittenBytes;
        }
    }

    /* Update GL_PRIMITIVES_GENERATED query by reading the tess factor buffer
     * and computing the number of primitives generated per patch.  The TES
     * compute kernel dispatch above only runs TES once per patch (not per
     * tessellated vertex), so we must manually compute the primitive count
     * that the hardware tessellator would have produced.
     *
     * MTLQuadTessellationFactorsHalf = { half edge[4]; half inside[2]; } = 12 B.
     * For triangles: primitives ≈ ceil(inside)² (rough estimate).
     * For quads:      primitives ≈ 2 × ceil(inside0) × ceil(inside1).
     * For isolines:   primitives ≈ ceil(edge[0]). */
    if (_tessellation.tessFactorBuffer) {
        const struct {
            uint16_t edge[4];
            uint16_t inside[2];
        } __attribute__((packed)) *tessFactors =
            (const void *)_tessellation.tessFactorBuffer.contents;

        GLenum genMode = tesProgram ? tesProgram->tess_gen_mode : GL_TRIANGLES;
        GLboolean pointMode = tesProgram ? tesProgram->tess_gen_point_mode : GL_FALSE;

        GLuint64 totalPrimitives = 0;
        for (GLuint p = 0; p < patchCount; p++) {
            /* Tessellation factors are half-floats.  Convert to float. */
            float edge[4], inside[2];
            for (int i = 0; i < 4; i++) {
                edge[i] = *(const __fp16 *)&tessFactors[p].edge[i];
                if (edge[i] < 1.0f) edge[i] = 1.0f;
            }
            for (int i = 0; i < 2; i++) {
                inside[i] = *(const __fp16 *)&tessFactors[p].inside[i];
                if (inside[i] < 1.0f) inside[i] = 1.0f;
            }

            GLuint perPatch = 0;
            if (pointMode) {
                /* Point mode: 1 primitive per tessellated point. */
                if (genMode == GL_QUADS) {
                    perPatch = (GLuint)(ceilf(inside[0]) * ceilf(inside[1]));
                } else if (genMode == GL_TRIANGLES) {
                    perPatch = (GLuint)(ceilf(inside[0]) * ceilf(inside[0]));
                } else { /* GL_ISOLINES */
                    perPatch = (GLuint)ceilf(edge[0]);
                }
            } else {
                if (genMode == GL_QUADS) {
                    /* Each quad splits into 2 triangles. */
                    perPatch = 2u * (GLuint)(ceilf(inside[0]) * ceilf(inside[1]));
                } else if (genMode == GL_TRIANGLES) {
                    perPatch = (GLuint)(ceilf(inside[0]) * ceilf(inside[0]));
                } else { /* GL_ISOLINES */
                    /* Each isoline segment is 1 line primitive (2 vertices). */
                    perPatch = (GLuint)ceilf(edge[0]);
                }
            }
            if (perPatch == 0u) perPatch = 1u;
            totalPrimitives += perPatch;
        }

        GLuint64 writtenPrimitives = totalPrimitives;
        if (xfbCaptureActive && writtenPrimitives > (GLuint64)xfbPrimitiveCapacity) {
            writtenPrimitives = (GLuint64)xfbPrimitiveCapacity;
        }
        mglRecordActivePrimitiveQueryDraw(glm_ctx, totalPrimitives, writtenPrimitives);
    }

    return true;
}

@end
