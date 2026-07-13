// MGLRenderer+Compute.m
// Compute dispatch methods extracted from MGLRenderer.m.
// P2-1: Split from MGLRenderer.m to reduce file size.
// These methods do not depend on any file-scope static functions in MGLRenderer.m.

#import "MGLRenderer_Private.h"

@implementation MGLRenderer (Compute)

#pragma mark ----- compute utility ---------------------------------------------------------------------

- (bool) bindBuffersToComputeEncoder:(id <MTLComputeCommandEncoder>) computeCommandEncoder
{
    if (!computeCommandEncoder) {
        NSLog(@"MGL COMPUTE ERROR: NULL compute encoder for buffer binding");
        return false;
    }

    RETURN_FALSE_ON_FAILURE([self mapGLBuffersToMTLBufferMap: &ctx->active_state->compute_buffer_map_list stage:_COMPUTE_SHADER]);

    // dirty buffer covers all buffer modifications
    if (ctx->active_state->dirty_bits & DIRTY_BUFFER)
    {
        // updateDirtyBaseBufferList binds new mtl buffers or updates old ones
        [self updateDirtyBaseBufferList: &ctx->active_state->compute_buffer_map_list];

        ctx->active_state->dirty_bits &= ~DIRTY_BUFFER;
    }

    for(int i=0; i<ctx->active_state->compute_buffer_map_list.count; i++)
    {
        BufferMap *map = &ctx->active_state->compute_buffer_map_list.buffers[i];
        Buffer *ptr;
        NSUInteger metalBindingIndex;
        NSUInteger bindOffset;

        ptr = map->buf;

        RETURN_FALSE_ON_NULL(ptr);
        RETURN_FALSE_ON_NULL(ptr->data.mtl_data);

        metalBindingIndex = map->has_metal_binding
            ? (NSUInteger)map->metal_binding_index
            : (NSUInteger)map->buffer_base_index;
        if (metalBindingIndex >= kMGLMaxMetalVertexBufferCount) {
            NSLog(@"MGL COMPUTE WARNING: buffer map[%d] Metal slot %lu out of range, skipping",
                  i,
                  (unsigned long)metalBindingIndex);
            continue;
        }
        if (map->offset < 0) {
            NSLog(@"MGL COMPUTE WARNING: buffer map[%d] negative offset=%lld, skipping",
                  i,
                  (long long)map->offset);
            continue;
        }
        bindOffset = (NSUInteger)map->offset;

        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(ptr->data.mtl_data);
        if (!buffer) {
            NSLog(@"MGL COMPUTE ERROR: buffer %u has NULL Metal buffer after mapping", ptr->name);
            return false;
        }
        if (bindOffset >= buffer.length) {
            NSLog(@"MGL COMPUTE WARNING: buffer map[%d] buffer=%u offset=%lu length=%lu, skipping",
                  i,
                  (unsigned)ptr->name,
                  (unsigned long)bindOffset,
                  (unsigned long)buffer.length);
            continue;
        }

        [computeCommandEncoder setBuffer:buffer offset:bindOffset atIndex:metalBindingIndex];
    }

    /* Bind spvBufferSizeConstants for runtime-sized SSBO arrays.
     * SPIRV-Cross emits code that reads uint32 byte-sizes from a
     * constant uint* buffer at MGL_BUFFER_SIZE_BUFFER_INDEX when a
     * shader uses .length() on unsized SSBO arrays. */
    {
        Program *computeProgram = mglResolveProgramForStageFromState(ctx, _COMPUTE_SHADER);
        if (computeProgram && computeProgram->spirv[_COMPUTE_SHADER].needs_buffer_size_buffer)
        {
            uint32_t sizeConstants[31];
            memset(sizeConstants, 0, sizeof(sizeConstants));

            for (int i = 0; i < ctx->active_state->compute_buffer_map_list.count; i++)
            {
                BufferMap *map = &ctx->active_state->compute_buffer_map_list.buffers[i];
                if (!map->buf)
                    continue;
                NSUInteger metalSlot = map->has_metal_binding
                    ? (NSUInteger)map->metal_binding_index
                    : (NSUInteger)map->buffer_base_index;
                if (metalSlot >= 31 || metalSlot == MGL_BUFFER_SIZE_BUFFER_INDEX)
                    continue;
                GLsizeiptr visibleSize = map->size > 0 ? map->size : (map->buf->size - map->offset);
                if (visibleSize < 0) visibleSize = 0;
                sizeConstants[metalSlot] = (uint32_t)visibleSize;
            }

            id<MTLBuffer> sizeBuffer = [_device newBufferWithBytes:sizeConstants
                                                                 length:sizeof(sizeConstants)
                                                                options:MTLResourceStorageModeShared];
            if (sizeBuffer) {
                [computeCommandEncoder setBuffer:sizeBuffer offset:0 atIndex:MGL_BUFFER_SIZE_BUFFER_INDEX];
            }
        }
    }

    return true;
}

- (bool) bindTexturesToComputeEncoder:(id <MTLComputeCommandEncoder>) computeCommandEncoder
{
    GLuint count;
    enum {
        _TEXTURE,
        _IMAGE_TEXTURE
    };
    struct {
        int spvc_type;
        int gl_texture_type;
    } mapped_types[] = {
        {SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, _TEXTURE},
        {SPVC_RESOURCE_TYPE_STORAGE_IMAGE, _IMAGE_TEXTURE},
        {0,0}
    };

    if (!computeCommandEncoder) {
        NSLog(@"MGL COMPUTE ERROR: NULL compute encoder for texture binding");
        return false;
    }

    Program *computeProgram = mglResolveProgramForStageFromState(ctx, _COMPUTE_SHADER);

    for(int type=0; mapped_types[type].spvc_type; type++)
    {
        int spvc_type;
        int gl_texture_type;

        spvc_type = mapped_types[type].spvc_type;
        gl_texture_type = mapped_types[type].gl_texture_type;

        // iterate shader storage buffers
        count = [self getProgramBindingCount: _COMPUTE_SHADER type: spvc_type];
        if (count)
        {
            int textures_to_be_mapped = count;

            if (textures_to_be_mapped > TEXTURE_UNITS) {
                textures_to_be_mapped = TEXTURE_UNITS;
            }

            for (int i=0; i < (int)count && textures_to_be_mapped > 0; i++)
            {
                SpirvResource *resource = NULL;
                GLuint metalBinding = [self getProgramBinding:_COMPUTE_SHADER type:spvc_type index:i];
                GLuint glUnit = 0u;
                Texture *ptr = NULL;

                if (computeProgram &&
                    spvc_type >= 0 && spvc_type < _MAX_SPIRV_RES &&
                    i >= 0 &&
                    i < (int)computeProgram->spirv_resources_list[_COMPUTE_SHADER][spvc_type].count) {
                    resource = &computeProgram->spirv_resources_list[_COMPUTE_SHADER][spvc_type].list[i];
                    metalBinding = mglMetalResourceSlot(resource);
                }

                if (metalBinding >= TEXTURE_UNITS ||
                    mglShouldSkipStageTextureResource(computeProgram,
                                                      _COMPUTE_SHADER,
                                                      spvc_type,
                                                      resource)) {
                    continue;
                }

                switch(gl_texture_type)
                {
                    case _TEXTURE:
                        glUnit = [self textureUnitForSampledResource:resource
                                                         metalBinding:metalBinding
                                                                stage:_COMPUTE_SHADER];
                        if (glUnit >= TEXTURE_UNITS) {
                            continue;
                        }
                        ptr = [self textureForSampledResource:resource
                                                 metalBinding:metalBinding
                                                         stage:_COMPUTE_SHADER
                                                  expectedType:[self getProgramDeclaredTextureType:_COMPUTE_SHADER
                                                                                              type:spvc_type
                                                                                             index:i]];
                        break;
                    case _IMAGE_TEXTURE:
                        glUnit = resource ? (resource->sampler_unit >= 0 ? (GLuint)resource->sampler_unit : resource->gl_binding)
                                          : [self getProgramGLBinding:_COMPUTE_SHADER
                                                                                        type:spvc_type
                                                                                       index:i];
                        if (glUnit >= TEXTURE_UNITS) {
                            continue;
                        }
                        ptr = STATE(image_units[glUnit].tex);
                        break;
                    default:
                        ptr = NULL;
                        NSLog(@"MGL COMPUTE ERROR: unknown compute texture binding class %d", gl_texture_type);
                        return false;
                }

                if (ptr)
                {
                    RETURN_FALSE_ON_FAILURE([self bindMTLTexture: ptr]);
                    if (!ptr->mtl_data) {
                        continue;
                    }

                    id<MTLTexture> texture;
                    texture = (__bridge id<MTLTexture>)(ptr->mtl_data);
                    if (!texture) {
                        continue;
                    }

                    /* For storage images bound to a non-zero mipmap level, create
                     * a level-specific texture view so imageSize() returns the
                     * dimensions at the bound level (matches the fragment-stage
                     * path).  Sampled textures are not affected. */
                    if (gl_texture_type == _IMAGE_TEXTURE) {
                        GLuint imgLevel = STATE(image_units[glUnit].level);
                        if (imgLevel > 0u) {
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

                    id<MTLSamplerState> sampler;

                    // late binding of texture samplers.. but its better than scanning the entire texture_samplers
                    if(gl_texture_type == _TEXTURE && STATE(texture_samplers[glUnit]))
                    {
                        Sampler *gl_sampler;

                        gl_sampler = STATE(texture_samplers[glUnit]);

                        // delete existing sampler if dirty
                        if (gl_sampler->dirty_bits)
                        {
                            if (gl_sampler->mtl_data)
                            {
                                mglSafeReleaseMetalObj((void **)&gl_sampler->mtl_data);
                            }
                        }

                        if (gl_sampler->mtl_data == NULL)
                        {
                            gl_sampler->mtl_data = (void *)CFBridgingRetain([self createMTLSamplerForTexParam:&gl_sampler->params target:ptr->target]);
                            gl_sampler->dirty_bits = 0;
                        }

                        sampler = (__bridge id<MTLSamplerState>)(gl_sampler->mtl_data);
                    }
                    else
                    {
                        sampler = (__bridge id<MTLSamplerState>)(ptr->params.mtl_data);
                    }

                    if (!sampler) {
                        id<MTLSamplerState> fallbackSampler = [_device newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]];
                        sampler = fallbackSampler;
                        if (!sampler) {
                            continue;
                        }
                    }

                    [computeCommandEncoder setTexture:texture atIndex:metalBinding];
                    if (gl_texture_type == _TEXTURE) {
                        [computeCommandEncoder setSamplerState:sampler atIndex:metalBinding];
                    }

                    textures_to_be_mapped--;
                }
            }

            // texture not found
            if (textures_to_be_mapped)
            {
                DEBUG_PRINT("No texture bound for fragment shader location\n");

                return false;
            }
        }
    }

    if (computeProgram) {
        SpirvResourceList *arrayResources =
            &computeProgram->spirv_resources_list[_COMPUTE_SHADER][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];
        for (GLuint resourceIndex = 0; arrayResources->list && resourceIndex < arrayResources->count; resourceIndex++) {
            SpirvResource *resource = &arrayResources->list[resourceIndex];
            if (resource->gl_array_size <= 1) {
                continue;
            }

            MTLTextureType expectedType =
                [self getProgramDeclaredTextureType:_COMPUTE_SHADER
                                               type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                              index:(int)resourceIndex];
            for (GLint element = 1; element < resource->gl_array_size; element++) {
                GLuint metalSlot = resource->binding + (GLuint)element;
                if (metalSlot >= TEXTURE_UNITS) {
                    break;
                }

                GLuint glUnit = [self textureUnitForSampledResource:NULL
                                                        metalBinding:metalSlot
                                                               stage:_COMPUTE_SHADER];
                Texture *ptr = [self textureForSampledResource:NULL
                                                   metalBinding:metalSlot
                                                           stage:_COMPUTE_SHADER
                                                    expectedType:expectedType];
                if (!ptr || ![self bindMTLTexture:ptr] || !ptr->mtl_data) {
                    continue;
                }

                id<MTLTexture> texture = (__bridge id<MTLTexture>)(ptr->mtl_data);
                id<MTLSamplerState> sampler = nil;
                if (glUnit < TEXTURE_UNITS && STATE(texture_samplers[glUnit])) {
                    Sampler *glSampler = STATE(texture_samplers[glUnit]);
                    if (glSampler->mtl_data == NULL) {
                        glSampler->mtl_data = (void *)CFBridgingRetain(
                            [self createMTLSamplerForTexParam:&glSampler->params target:ptr->target]);
                        glSampler->dirty_bits = 0;
                    }
                    sampler = (__bridge id<MTLSamplerState>)(glSampler->mtl_data);
                } else if (ptr->params.mtl_data) {
                    sampler = (__bridge id<MTLSamplerState>)(ptr->params.mtl_data);
                }
                if (!sampler) {
                    sampler = [_device newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]];
                }

                [computeCommandEncoder setTexture:texture atIndex:metalSlot];
                if (sampler) {
                    [computeCommandEncoder setSamplerState:sampler atIndex:metalSlot];
                }
            }
        }
    }

    /* Bind additional array elements for storage image arrays.
     * SPIRV-Cross emits `array<texture2d<T, access::read_write>, N> image [[texture(B)]]`
     * which occupies consecutive Metal texture slots B..B+N-1.  The main
     * loop above only binds element 0; bind elements 1..N-1 here. */
    if (computeProgram) {
        SpirvResourceList *storageArrayResources =
            &computeProgram->spirv_resources_list[_COMPUTE_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE];
        for (GLuint resourceIndex = 0; storageArrayResources->list && resourceIndex < storageArrayResources->count; resourceIndex++) {
            SpirvResource *resource = &storageArrayResources->list[resourceIndex];
            if (resource->gl_array_size <= 1) {
                continue;
            }

            for (GLint element = 1; element < resource->gl_array_size; element++) {
                GLuint metalSlot = resource->binding + (GLuint)element;
                if (metalSlot >= TEXTURE_UNITS) {
                    break;
                }

                GLuint glUnit = (resource->sampler_unit >= 0 ? (GLuint)resource->sampler_unit : resource->gl_binding) + (GLuint)element;
                if (glUnit >= TEXTURE_UNITS) {
                    continue;
                }

                Texture *ptr = STATE(image_units[glUnit].tex);
                if (!ptr || ![self bindMTLTexture:ptr] || !ptr->mtl_data) {
                    continue;
                }

                id<MTLTexture> texture = (__bridge id<MTLTexture>)(ptr->mtl_data);

                /* For storage images bound to a non-zero mipmap level, create
                 * a level-specific texture view (matches element 0 path). */
                GLuint imgLevel = STATE(image_units[glUnit].level);
                if (imgLevel > 0u) {
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

                [computeCommandEncoder setTexture:texture atIndex:metalSlot];
            }
        }
    }

    ctx->active_state->dirty_bits &= ~(DIRTY_TEX_BINDING | DIRTY_SAMPLER | DIRTY_IMAGE_UNIT_STATE);

    return true;
}

#pragma mark ------------------------------------------------------------------------------------------
#pragma mark processCompute
#pragma mark ------------------------------------------------------------------------------------------
-(bool)processCompute:(id <MTLComputeCommandEncoder>) computeCommandEncoder
{
    // from https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Compute-Ctx/Compute-Ctx.html#//apple_ref/doc/uid/TP40014221-CH6-SW1
    Program *program;

    if (!computeCommandEncoder) {
        NSLog(@"MGL COMPUTE ERROR: processCompute called with NULL encoder");
        return false;
    }

    program = mglResolveProgramForStageFromState(ctx, _COMPUTE_SHADER);
    if (!program) {
        NSLog(@"MGL COMPUTE ERROR: glDispatchCompute with no current program");
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return false;
    }

    if (program->dirty_bits)
    {
        if (![self bindMTLProgram: program]) {
            NSLog(@"MGL COMPUTE ERROR: failed to bind compute program %u", program->name);
            return false;
        }
    }

    Shader *computeShader;
    computeShader = program->shader_slots[_COMPUTE_SHADER];
    if (!computeShader) {
        NSLog(@"MGL COMPUTE ERROR: current program %u has no compute shader", program->name);
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return false;
    }

    id <MTLFunction> func;
    func = (__bridge id<MTLFunction>)(computeShader->mtl_data.function);
    if (!func) {
        NSLog(@"MGL COMPUTE ERROR: compute shader for program %u has no Metal function", program->name);
        return false;
    }

    id <MTLComputePipelineState> computePipelineState;
    NSError *errors;
    computePipelineState = [_device newComputePipelineStateWithFunction:func error: &errors];
    if (!computePipelineState) {
        NSLog(@"MGL COMPUTE ERROR: failed to create compute pipeline for program %u: %@",
              program->name,
              errors);
        return false;
    }

    [computeCommandEncoder setComputePipelineState:computePipelineState];

    RETURN_FALSE_ON_FAILURE([self bindBuffersToComputeEncoder: computeCommandEncoder]);

    //setTexture:atIndex:
    //setTextures:withRange:
    RETURN_FALSE_ON_FAILURE([self bindTexturesToComputeEncoder: computeCommandEncoder]);

    // setSamplerState:atIndex:
    // setSamplerState:lodMinClamp:lodMaxClamp:atIndex:
    // setSamplerStates:withRange:
    // setSamplerStates:lodMinClamps:lodMaxClamps:withRange:

    // [computeCommandEncoder setThreadgroupMemoryLength:atIndex:

    ctx->active_state->dirty_bits = 0;

    return true;
}

-(void)mtlDispatchCompute:(GLMContext)glm_ctx groupsX:(GLuint)groups_x groupsY:(GLuint)groups_y groupsZ:(GLuint)groups_z
{
    if (!glm_ctx) {
        NSLog(@"MGL COMPUTE ERROR: mtlDispatchCompute called with NULL context");
        return;
    }

    ctx = glm_ctx;

    if (groups_x == 0 || groups_y == 0 || groups_z == 0) {
        NSLog(@"MGL COMPUTE TRACE: glDispatchCompute zero-sized dispatch %ux%ux%u skipped",
              groups_x,
              groups_y,
              groups_z);
        return;
    }

    // end encoding on current render encoder
    [self endRenderEncoding];

    RETURN_ON_FAILURE([self ensureWritableCommandBuffer:"mtlDispatchCompute"]);

    for (NSUInteger unit = 0; unit < TEXTURE_UNITS; unit++) {
        Texture *imageTexture = glm_ctx->active_state->image_units[unit].tex;
        if (imageTexture) {
            RETURN_ON_FAILURE([self bindMTLTexture:imageTexture]);
        }

        Texture *sampledTexture = glm_ctx->active_state->active_textures[unit];
        if (sampledTexture) {
            RETURN_ON_FAILURE([self bindMTLTexture:sampledTexture]);
        }
    }

    id <MTLComputeCommandEncoder> computeCommandEncoder = [_currentCommandBuffer computeCommandEncoder];
    if (!computeCommandEncoder) {
        NSLog(@"MGL ERROR: Failed to create compute command encoder");
        return;
    }

    if (![self processCompute:computeCommandEncoder]) {
        [computeCommandEncoder endEncoding];
        return;
    }

    MTLSize numThreadgroups;
    MTLSize threadsPerThreadgroup;

    Program *ptr;
    ptr = mglResolveProgramForStageFromState(glm_ctx, _COMPUTE_SHADER);
    if (!ptr) {
        NSLog(@"MGL COMPUTE ERROR: glDispatchCompute with no current compute program after binding");
        [computeCommandEncoder endEncoding];
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    GLuint local_x = ptr->local_workgroup_size.x ? ptr->local_workgroup_size.x : 1u;
    GLuint local_y = ptr->local_workgroup_size.y ? ptr->local_workgroup_size.y : 1u;
    GLuint local_z = ptr->local_workgroup_size.z ? ptr->local_workgroup_size.z : 1u;

    if (ptr->local_workgroup_size.x || ptr->local_workgroup_size.y || ptr->local_workgroup_size.z)
    {
        numThreadgroups = MTLSizeMake(groups_x, groups_y, groups_z);
        threadsPerThreadgroup = MTLSizeMake(local_x, local_y, local_z);

        [computeCommandEncoder dispatchThreadgroups:numThreadgroups
                                        threadsPerThreadgroup:threadsPerThreadgroup];
    }
    else
    {
        numThreadgroups = MTLSizeMake(groups_x, groups_y, groups_z);
        threadsPerThreadgroup = MTLSizeMake(1, 1, 1);

        [computeCommandEncoder dispatchThreadgroups:numThreadgroups
                                        threadsPerThreadgroup:threadsPerThreadgroup];
    }

    [computeCommandEncoder endEncoding];

    for (NSUInteger unit = 0; unit < TEXTURE_UNITS; unit++) {
        ImageUnit *imageUnit = &glm_ctx->active_state->image_units[unit];
        Texture *imageTexture = imageUnit->tex;
        if (!imageTexture ||
            (imageUnit->access != GL_WRITE_ONLY && imageUnit->access != GL_READ_WRITE)) {
            continue;
        }
        imageTexture->metal_data_authoritative = GL_TRUE;
        if (imageTexture->faces[0].levels &&
            imageUnit->level >= 0 &&
            imageUnit->level < (GLint)imageTexture->num_levels) {
            imageTexture->faces[0].levels[imageUnit->level].metal_data_authoritative = GL_TRUE;
        }
    }

    /* P2-6: Fine-grained dirty bits instead of DIRTY_ALL.  Compute dispatch
     * ends the render encoder, so the next draw must rebuild it.  DIRTY_STATE
     * triggers newRenderEncoderLocked; DIRTY_FBO re-syncs the render pass;
     * the remaining bits (matching kMGLFullReplayDirtyBits in MGLRenderer+Draw.m)
     * re-bind all GL resources that the render encoder needs.  DIRTY_SHADER and
     * DIRTY_DRAWABLE are intentionally excluded — DIRTY_SHADER is a per-program
     * bit, and DIRTY_DRAWABLE only applies at context init. */
    glm_ctx->active_state->dirty_bits |=
        (DIRTY_STATE | DIRTY_FBO | DIRTY_PROGRAM | DIRTY_VAO |
         DIRTY_RENDER_STATE | DIRTY_TEX_BINDING | DIRTY_TEX |
         DIRTY_TEX_PARAM | DIRTY_SAMPLER | DIRTY_ALPHA_STATE |
         DIRTY_BUFFER | DIRTY_BUFFER_BASE_STATE | DIRTY_IMAGE_UNIT_STATE);

    //[self newRenderEncoder];
}


-(void)mtlDispatchComputeIndirect:(GLMContext)glm_ctx indirect:(GLintptr)indirect
{
    if (!glm_ctx) {
        NSLog(@"MGL COMPUTE ERROR: mtlDispatchComputeIndirect called with NULL context");
        return;
    }

    ctx = glm_ctx;

    Buffer *glIndirectBuffer = glm_ctx->active_state->buffers[_DISPATCH_INDIRECT_BUFFER];
    if (glm_ctx->active_state->var.dispatch_indirect_buffer_binding == 0 || !glIndirectBuffer) {
        NSLog(@"MGL COMPUTE ERROR: glDispatchComputeIndirect with no GL_DISPATCH_INDIRECT_BUFFER bound");
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }
    if (indirect < 0) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_VALUE);
        return;
    }

    if (![self processBuffer:glIndirectBuffer]) {
        NSLog(@"MGL COMPUTE ERROR: failed to process dispatch indirect buffer %u",
              glIndirectBuffer ? glIndirectBuffer->name : 0u);
        return;
    }

    id<MTLBuffer> indirectBuffer = (__bridge id<MTLBuffer>)(glIndirectBuffer->data.mtl_data);
    if (!indirectBuffer) {
        NSLog(@"MGL COMPUTE ERROR: dispatch indirect buffer %u has no Metal backing",
              glIndirectBuffer ? glIndirectBuffer->name : 0u);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    NSUInteger indirectOffset = (NSUInteger)indirect;
    NSUInteger indirectArgBytes = 3u * sizeof(uint32_t);
    if (indirectOffset > indirectBuffer.length ||
        indirectArgBytes > (indirectBuffer.length - indirectOffset)) {
        NSLog(@"MGL COMPUTE ERROR: dispatch indirect range exceeds Metal buffer buffer=%u off=%lu bytes=%lu len=%lu",
              glIndirectBuffer ? glIndirectBuffer->name : 0u,
              (unsigned long)indirectOffset,
              (unsigned long)indirectArgBytes,
              (unsigned long)indirectBuffer.length);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    [self endRenderEncoding];

    RETURN_ON_FAILURE([self ensureWritableCommandBuffer:"mtlDispatchComputeIndirect"]);

    for (NSUInteger unit = 0; unit < TEXTURE_UNITS; unit++) {
        Texture *imageTexture = glm_ctx->active_state->image_units[unit].tex;
        if (imageTexture) {
            RETURN_ON_FAILURE([self bindMTLTexture:imageTexture]);
        }

        Texture *sampledTexture = glm_ctx->active_state->active_textures[unit];
        if (sampledTexture) {
            RETURN_ON_FAILURE([self bindMTLTexture:sampledTexture]);
        }
    }

    id<MTLComputeCommandEncoder> computeCommandEncoder = [_currentCommandBuffer computeCommandEncoder];
    if (!computeCommandEncoder) {
        NSLog(@"MGL ERROR: Failed to create compute command encoder for indirect dispatch");
        return;
    }

    if (![self processCompute:computeCommandEncoder]) {
        [computeCommandEncoder endEncoding];
        return;
    }

    Program *ptr = mglResolveProgramForStageFromState(glm_ctx, _COMPUTE_SHADER);
    if (!ptr) {
        NSLog(@"MGL COMPUTE ERROR: glDispatchComputeIndirect with no current compute program after binding");
        [computeCommandEncoder endEncoding];
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    GLuint local_x = ptr->local_workgroup_size.x ? ptr->local_workgroup_size.x : 1u;
    GLuint local_y = ptr->local_workgroup_size.y ? ptr->local_workgroup_size.y : 1u;
    GLuint local_z = ptr->local_workgroup_size.z ? ptr->local_workgroup_size.z : 1u;
    MTLSize threadsPerThreadgroup = MTLSizeMake(local_x, local_y, local_z);

    [computeCommandEncoder dispatchThreadgroupsWithIndirectBuffer:indirectBuffer
                                             indirectBufferOffset:indirectOffset
                                            threadsPerThreadgroup:threadsPerThreadgroup];

    [computeCommandEncoder endEncoding];

    /* P2-6: Fine-grained dirty bits — see mtlDispatchCompute for rationale. */
    glm_ctx->active_state->dirty_bits |=
        (DIRTY_STATE | DIRTY_FBO | DIRTY_PROGRAM | DIRTY_VAO |
         DIRTY_RENDER_STATE | DIRTY_TEX_BINDING | DIRTY_TEX |
         DIRTY_TEX_PARAM | DIRTY_SAMPLER | DIRTY_ALPHA_STATE |
         DIRTY_BUFFER | DIRTY_BUFFER_BASE_STATE | DIRTY_IMAGE_UNIT_STATE);
}

@end
