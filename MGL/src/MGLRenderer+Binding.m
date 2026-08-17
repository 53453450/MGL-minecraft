// MGLRenderer+Binding.m
// Buffer/texture Metal object binding methods extracted from MGLRenderer+RenderPass.m

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Blit_Private.h"
#include "mgl_render_cpp.h"

static BOOL mglBindingUsesMetalCpp(void)
{
    return mglRenderCppGetDevice() != NULL;
}

void mglRendererCallbackBindTexture(void *runtime_context,
                                    GLMContext glm_ctx,
                                    Texture *texture)
{
    MGLRenderer *renderer = (__bridge MGLRenderer *)runtime_context;
    if (!renderer || !glm_ctx || !texture) return;
    (void)[renderer bindMTLTexture:texture];
}

static MGLMetalBufferRef mglBindingCreateBuffer(MGLMetalDeviceRef device,
                                            NSUInteger length,
                                            MTLResourceOptions options)
{
    if (mglBindingUsesMetalCpp()) {
        void *buffer = NULL;
        if (mglRenderCppCreateBuffer(length, options, NULL, &buffer) == 0 &&
            buffer) {
            return (__bridge_transfer MGLMetalBufferRef)buffer;
        }
    }
    return [device newBufferWithLength:length options:options];
}

static MGLMetalBufferRef mglBindingCreateBufferWithBytes(
    MGLMetalDeviceRef device,
    const void *bytes,
    NSUInteger length,
    MTLResourceOptions options)
{
    if (mglBindingUsesMetalCpp()) {
        void *buffer = NULL;
        if (mglRenderCppCreateBufferWithBytes(bytes, length, options, NULL,
                                              &buffer) == 0 && buffer) {
            return (__bridge_transfer MGLMetalBufferRef)buffer;
        }
    }
    return [device newBufferWithBytes:bytes length:length options:options];
}

static MGLMetalBufferRef mglBindingCreateBufferWithBytesNoCopy(
    MGLMetalDeviceRef device,
    const void *bytes,
    NSUInteger length,
    MTLResourceOptions options,
    BOOL deallocateVM)
{
    if (mglBindingUsesMetalCpp()) {
        void *buffer = NULL;
        if (mglRenderCppCreateBufferWithBytesNoCopy(
                bytes, length, options, NULL, deallocateVM ? 1 : 0,
                &buffer) == 0 && buffer) {
            return (__bridge_transfer MGLMetalBufferRef)buffer;
        }
    }
    return [device newBufferWithBytesNoCopy:(void *)bytes
                                      length:length
                                     options:options
                                 deallocator:deallocateVM ? ^(void *pointer, NSUInteger size) {
        kern_return_t result = vm_deallocate((vm_map_t)mach_task_self(),
                                             (vm_address_t)pointer,
                                             (vm_size_t)size);
        if (result != KERN_SUCCESS) {
            NSLog(@"MGL WARNING: vm_deallocate failed for Metal no-copy buffer err=%d ptr=%p len=%lu",
                  result, pointer, (unsigned long)size);
        }
    } : nil];
}

static MGLMetalSamplerStateRef mglBindingCreateSampler(
    MGLMetalDeviceRef device,
    MTLSamplerDescriptor *descriptor)
{
    if (mglBindingUsesMetalCpp()) {
        void *sampler = NULL;
        if (mglRenderCppCreateSampler((__bridge void *)descriptor,
                                      &sampler) == 0 && sampler) {
            return (__bridge_transfer MGLMetalSamplerStateRef)sampler;
        }
    }
    return [device newSamplerStateWithDescriptor:descriptor];
}

static void mglBindingCopyTexture(MGLMetalBlitCommandEncoderRef encoder,
                                  MGLMetalTextureRef source,
                                  NSUInteger sourceSlice,
                                  NSUInteger sourceLevel,
                                  MTLOrigin sourceOrigin,
                                  MTLSize sourceSize,
                                  MGLMetalTextureRef destination,
                                  NSUInteger destinationSlice,
                                  NSUInteger destinationLevel,
                                  MTLOrigin destinationOrigin)
{
    [encoder copyFromTexture:source
                 sourceSlice:sourceSlice
                 sourceLevel:sourceLevel
                sourceOrigin:sourceOrigin
                  sourceSize:sourceSize
                   toTexture:destination
            destinationSlice:destinationSlice
            destinationLevel:destinationLevel
           destinationOrigin:destinationOrigin];
}

static void mglBindingEndBlitEncoder(MGLMetalBlitCommandEncoderRef encoder)
{
    [encoder endEncoding];
}

@implementation MGLRenderer (Binding)

- (void) bindMTLBuffer:(Buffer *) ptr
{
    METAL_LOCK();
    [self bindMTLBufferLocked:ptr];
    METAL_UNLOCK();
}

- (void) bindMTLBufferLocked:(Buffer *) ptr
{
    MTLResourceOptions options;
    const size_t kMaxSafeBufferSize = (size_t)2 * 1024 * 1024 * 1024; // 2 GiB safety cap

    if (mglBindingUsesMetalCpp()) {
        char bindError[256] = {0};
        int bindResult = mglRenderCppBindBufferStorage(
            ptr, bindError, sizeof(bindError));
        if (bindResult == MGL_RENDER_CPP_BUFFER_BOUND) {
            return;
        }
        if (bindResult == MGL_RENDER_CPP_BUFFER_ERROR) {
            NSLog(@"MGL ERROR: Metal-cpp buffer bind failed buffer=%u: %s",
                  ptr ? (unsigned)ptr->name : 0u,
                  bindError[0] ? bindError : "?");
            return;
        }
    }

    mglMetalCountCreate(MGLMetalKindBuffer);

    if (!ptr) {
        NSLog(@"MGL ERROR: bindMTLBuffer called with NULL buffer");
        return;
    }

    // Corrupted buffer sizes can crash Metal validation immediately.
    if (ptr->size == 0 || ptr->size > kMaxSafeBufferSize) {
        NSLog(@"MGL ERROR: Refusing to create Metal buffer with suspicious size=%zu for buffer %u",
              (size_t)ptr->size, ptr->name);
        ptr->data.mtl_data = NULL;
        return;
    }

    options = MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared;

    // ways we will only write to this
    if ((ptr->storage_flags & GL_MAP_READ_BIT) == 0)
    {
        options |= MTLResourceCPUCacheModeWriteCombined;
    }

    if (ptr->transient_batch_buffer)
    {
        if (!ptr->data.buffer_data) {
            NSLog(@"MGL ERROR: transient batch buffer has no CPU backing (size=%zu)", (size_t)ptr->size);
            ptr->data.mtl_data = NULL;
            return;
        }

        MGLMetalBufferRef buffer = mglBindingCreateBufferWithBytes(
            _device, (void *)(uintptr_t)ptr->data.buffer_data,
            (NSUInteger)ptr->size, options);
        if (!buffer) {
            NSLog(@"MGL ERROR: Failed to create transient batch Metal buffer (size=%zu)", (size_t)ptr->size);
            ptr->data.mtl_data = NULL;
            return;
        }

        ptr->data.mtl_data = (void *)CFBridgingRetain(buffer);
        return;
    }

    if (ptr->storage_flags & GL_CLIENT_STORAGE_BIT)
    {
        if (!ptr->data.buffer_data) {
            NSLog(@"MGL ERROR: GL_CLIENT_STORAGE_BIT set but buffer_data is NULL for buffer %u", ptr->name);
            ptr->data.mtl_data = NULL;
            return;
        }

        MGLMetalBufferRef buffer = mglBindingCreateBufferWithBytesNoCopy(
            _device, (void *)(ptr->data.buffer_data), ptr->size, options, YES);

        ptr->data.mtl_data = (void *)CFBridgingRetain(buffer);
        ptr->data.mtl_owns_buffer_data = buffer ? GL_TRUE : GL_FALSE;
    }
    else
    {
        MGLMetalBufferRef buffer;
        
        if (ptr->data.buffer_data)
        {
            size_t safeBufferSize = ptr->data.buffer_size;
            if (safeBufferSize == 0 || safeBufferSize > kMaxSafeBufferSize) {
                safeBufferSize = ptr->size;
            }

            if ((ptr->immutable_storage & BUFFER_IMMUTABLE_STORAGE_FLAG) &&
                (ptr->storage_flags & GL_MAP_PERSISTENT_BIT)) {
                /*
                 * A persistent GL mapping must keep returning the same CPU
                 * address.  Wrap that VM allocation directly so explicit
                 * range flushes update the Metal resource in place instead of
                 * allocating and copying a full-buffer snapshot per flush.
                 * The Metal object owns the VM range so in-flight command
                 * buffers keep it alive after GL deletion or context teardown.
                 */
                buffer = mglBindingCreateBufferWithBytesNoCopy(
                    _device, (void *)ptr->data.buffer_data,
                    safeBufferSize, options, YES);
                ptr->data.mtl_owns_buffer_data = buffer ? GL_TRUE : GL_FALSE;
            } else {
                buffer = mglBindingCreateBufferWithBytes(
                    _device, (void *)ptr->data.buffer_data,
                    safeBufferSize, options);
                ptr->data.mtl_owns_buffer_data = GL_FALSE;
            }
            if (!buffer) {
                NSLog(@"MGL ERROR: Failed to create Metal buffer from CPU backing (size=%zu, buffer=%u)",
                      safeBufferSize, ptr->name);
                ptr->data.mtl_data = NULL;
                return;
            }
        }
        else
        {
            buffer = mglBindingCreateBuffer(_device, ptr->size, options);
            if (!buffer) {
                NSLog(@"MGL ERROR: Failed to allocate Metal buffer with length=%zu (buffer=%u)",
                      (size_t)ptr->size, ptr->name);
                ptr->data.mtl_data = NULL;
                return;
            }

            ptr->data.buffer_data = (vm_address_t)NULL;
        }

        ptr->data.mtl_data = (void *)CFBridgingRetain(buffer);
        if ((ptr->storage_flags & GL_MAP_PERSISTENT_BIT) &&
            (ptr->data.dirty_bits & DIRTY_BUFFER_DATA) &&
            buffer.length > 0 &&
            buffer.storageMode == MTLStorageModeManaged) {
            [buffer didModifyRange:NSMakeRange(0, buffer.length)];
        }
    }
}

- (bool)bindMTLTexture:(Texture *)tex
{
    METAL_LOCK();
    bool result = [self bindMTLTextureLocked:tex];
    METAL_UNLOCK();
    return result;
}

- (bool)bindMTLTextureLocked:(Texture *)tex
{
    if (tex && tex->target == GL_TEXTURE_BUFFER && tex->texture_buffer &&
        tex->texture_buffer->data.dirty_bits) {
        tex->dirty_bits |= DIRTY_TEXTURE_DATA;
    }

    // If this texture is now used as a render target but was previously created
    // without render-target usage, force a recreate with proper usage flags.
    // When the old texture already has GPU-written data (e.g. from imageStore
    // in a compute shader), preserve it via a GPU-to-GPU blit instead of
    // re-uploading potentially stale CPU data.
    if (tex->mtl_data && tex->is_render_target) {
        MGLMetalTextureRef existingTexture = (__bridge MGLMetalTextureRef)(tex->mtl_data);
        MTLTextureUsage requiredRenderTargetUsage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        NSUInteger requiredMipLevels =
            (tex->target == GL_RENDERBUFFER || tex->samples > 1u)
                ? 1u
                : ((tex->mipmap_levels > 1u) ? (NSUInteger)tex->mipmap_levels : 1u);
        BOOL usageMismatch = existingTexture &&
            ((existingTexture.usage & requiredRenderTargetUsage) != requiredRenderTargetUsage);
        BOOL mipCountMismatch = existingTexture &&
            requiredMipLevels > existingTexture.mipmapLevelCount;
        if (existingTexture && (usageMismatch || mipCountMismatch)) {
            NSLog(@"MGL WARNING: Recreating texture %u for render-target use (old usage=0x%lx oldMips=%lu requiredMips=%lu)",
                  tex->name,
                  (unsigned long)existingTexture.usage,
                  (unsigned long)existingTexture.mipmapLevelCount,
                  (unsigned long)requiredMipLevels);

            // Keep a strong reference to the old texture so we can blit its GPU
            // data to the new one after releasing tex->mtl_data.
            __strong MGLMetalTextureRef oldTexture = existingTexture;

            mglSafeReleaseMetalObj((void **)&tex->mtl_data);
            [self releaseGLSampledRenderTargetCopyForTexture:tex];

            // Create a new texture with correct usage.  Don't set
            // DIRTY_TEXTURE_DATA so that createMTLTextureFromGLTexture
            // skips CPU data upload — we'll blit GPU data instead.
            MGLMetalTextureRef newTexture = [self createMTLTextureFromGLTexture:tex];
            if (newTexture && oldTexture &&
                newTexture.width == oldTexture.width &&
                newTexture.height == oldTexture.height &&
                newTexture.depth == oldTexture.depth) {
                // Blit GPU data from old texture to new texture to preserve
                // any writes (e.g. imageStore) that occurred before the
                // is_render_target transition.
                [self endRenderEncodingLocked];
                if ([self ensureWritableCommandBufferLocked:"is_render_target_blit"]) {
                    if (mglBindingUsesMetalCpp()) {
                        if (mglRenderCppCopyMatchingTextureSubresourcesForCommandBufferOwner(
                                _renderPassManager.state->currentCommandBufferOwner,
                                (__bridge void *)oldTexture,
                                (__bridge void *)newTexture) != 0) {
                            NSLog(@"MGL ERROR: Metal-cpp render-target preservation blit failed texture=%u",
                                  tex->name);
                            tex->dirty_bits |= DIRTY_TEXTURE_DATA;
                            return false;
                        }
                    } else {
                        MGLMetalBlitCommandEncoderRef blit =
                            mglRenderCreateBlitEncoderForCommandBufferOwner(
                                _renderPassManager.state->currentCommandBufferOwner);
                        if (!blit) {
                            tex->dirty_bits |= DIRTY_TEXTURE_DATA;
                            return false;
                        }
                        NSUInteger copySlices = MIN(oldTexture.arrayLength, newTexture.arrayLength);
                        NSUInteger copyLevels = MIN(oldTexture.mipmapLevelCount, newTexture.mipmapLevelCount);
                        for (NSUInteger slice = 0; slice < copySlices; slice++) {
                            for (NSUInteger level = 0; level < copyLevels; level++) {
                                NSUInteger lw = (oldTexture.width >> level);
                                NSUInteger lh = (oldTexture.height >> level);
                                if (lw == 0 || lh == 0) continue;
                                MTLSize levelSize = MTLSizeMake(lw, lh, 1);
                                mglBindingCopyTexture(
                                    blit, oldTexture, slice, level,
                                    MTLOriginMake(0, 0, 0), levelSize,
                                    newTexture, slice, level,
                                    MTLOriginMake(0, 0, 0));
                            }
                        }
                        mglBindingEndBlitEncoder(blit);
                    }
                }
                tex->mtl_data = (void *)CFBridgingRetain(newTexture);
                tex->dirty_bits = 0;
            } else {
                // Fallback: use the old CPU-data-upload path
                tex->dirty_bits |= DIRTY_TEXTURE_DATA;
            }
        }
    }

    if (tex->dirty_bits)
    {
        bool textureNeedsRebuild =
            (tex->dirty_bits & (DIRTY_TEXTURE_LEVEL | DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_ACCESS)) != 0;
        bool samplerNeedsRebuild =
            textureNeedsRebuild || ((tex->dirty_bits & DIRTY_TEXTURE_PARAM) != 0);
        bool storageShapeChanged =
            (tex->dirty_bits & (DIRTY_TEXTURE_LEVEL | DIRTY_TEXTURE_ACCESS)) != 0;

        if (tex->mtl_data &&
            !storageShapeChanged &&
            (tex->dirty_bits & DIRTY_TEXTURE_DATA) != 0) {
            MGLMetalTextureRef existingTexture = (__bridge MGLMetalTextureRef)(tex->mtl_data);
            if (existingTexture &&
                [self uploadFullCPUTextureDataIntoTexture:tex
                                                     metal:existingTexture
                                                    reason:"bindMTLTexture.dirtyData"]) {
                textureNeedsRebuild =
                    (tex->dirty_bits & (DIRTY_TEXTURE_LEVEL | DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_ACCESS)) != 0;
                samplerNeedsRebuild =
                    textureNeedsRebuild || ((tex->dirty_bits & DIRTY_TEXTURE_PARAM) != 0);
            }
        }

        // Texture parameter changes only affect the Metal sampler object. Do
        // not throw away texture storage for wrap/filter/lod updates; doing so
        // can turn Minecraft's frequent sampler changes into render-pass and
        // upload storms.
        if (tex->mtl_data)
        {
            if (textureNeedsRebuild) {
                mglSafeReleaseMetalObj((void **)&tex->mtl_data);
                [self releaseGLSampledRenderTargetCopyForTexture:tex];
            }
        }

        if (samplerNeedsRebuild && tex->params.mtl_data)
        {
            mglSafeReleaseMetalObj((void **)&tex->params.mtl_data);
        }
    }

    if (tex->mtl_data == NULL)
    {
        tex->mtl_data = (void *)CFBridgingRetain([self createMTLTextureFromGLTexture: tex]);

        // AGX-SAFE: Handle NULL texture gracefully when in GPU recovery mode
        if (!tex->mtl_data) {
            // Circuit breaker: limit fallback texture creations to prevent infinite loops
            static int s_fallbackTextureCount = 0;
            static NSTimeInterval s_fallbackTextureWindowStart = 0;
            NSTimeInterval now = [[NSDate date] timeIntervalSince1970];
            if (now - s_fallbackTextureWindowStart > 5.0) {
                s_fallbackTextureCount = 0;
                s_fallbackTextureWindowStart = now;
            }
            if (s_fallbackTextureCount >= 4096) {
                NSLog(@"MGL AGX: Fallback texture limit reached (%d in %.1fs), suppressing further fallbacks",
                      s_fallbackTextureCount, now - s_fallbackTextureWindowStart);
                tex->mtl_data = NULL;
                tex->dirty_bits = 0;
            } else {
                s_fallbackTextureCount++;
                NSLog(@"MGL AGX: Primary texture creation returned NULL, attempting fallback texture creation (%d/4096)",
                      s_fallbackTextureCount);
                // Create a simple fallback texture to prevent crashes
                tex->mtl_data = (void *)CFBridgingRetain([self createFallbackMTLTexture: tex]);

                if (tex->mtl_data) {
                    NSLog(@"MGL SUCCESS: Fallback texture created successfully");
                    tex->dirty_bits = 0;
                } else {
                    NSLog(@"MGL ERROR: Even fallback texture creation failed - this texture will remain NULL");
                }
            }
        } else {
            if (kMGLDiagnosticStateLogs) {
                mglTraceLogNSString(@"MGL SUCCESS: Primary texture created successfully");
            }
        }

    }

    if (tex->params.mtl_data == NULL)
    {
        tex->params.mtl_data = (void *)CFBridgingRetain([self createMTLSamplerForTexParam:&tex->params target:tex->target]);
        // Sampler creation should not fail even in recovery mode
        if (!tex->params.mtl_data) {
            NSLog(@"MGL WARNING: Sampler creation failed, using default");
            tex->params.mtl_data = (void *)CFBridgingRetain(
                mglBindingCreateSampler(_device, [MTLSamplerDescriptor new]));
        }
        if ((tex->name == 21u || tex->name == 27u) &&
            mglEnvFlagEnabled("MGL_TRACE_TEXTURE_NAMES")) {
            mglTraceLogExternal("SAMPLER_CREATE tex=%u minFilter=0x%x magFilter=0x%x mipFilter=%d minLod=%.3f maxLod=%.3f base=%u max=%u mips=%u",
                                (unsigned)tex->name,
                                (unsigned)tex->params.min_filter,
                                (unsigned)tex->params.mag_filter,
                                (tex->params.min_filter >= 0x2700) ? 1 : 0,
                                (double)tex->params.min_lod,
                                (double)tex->params.max_lod,
                                (unsigned)tex->params.base_level,
                                (unsigned)tex->params.max_level,
                                (unsigned)tex->mipmap_levels);
        }
    }

    if (tex->params.mtl_data) {
        tex->dirty_bits &= ~DIRTY_TEXTURE_PARAM;
    }

    if (mglMipDiagEnabled()) {
        MGLMetalTextureRef mtlTex = (__bridge MGLMetalTextureRef)(tex->mtl_data);
        uint64_t signature = 1469598103934665603ULL;
        signature = mglMipDiagMixState(signature, (uint64_t)(uintptr_t)tex->mtl_data);
        signature = mglMipDiagMixState(signature, mtlTex ? mtlTex.mipmapLevelCount : 0u);
        signature = mglMipDiagMixState(signature, tex->num_levels);
        signature = mglMipDiagMixState(signature, tex->mipmap_levels);
        signature = mglMipDiagMixState(signature, tex->params.base_level);
        signature = mglMipDiagMixState(signature, tex->params.max_level);
        signature = mglMipDiagMixState(signature, tex->mipmapped ? 1u : 0u);
        signature = mglMipDiagMixState(signature, tex->genmipmaps ? 1u : 0u);

        /* Direct-mapped by name; a collision only costs an extra line. */
        static uint64_t s_textureState[128];
        if (mglMipDiagStateChanged(&s_textureState[tex->name & 127u], signature)) {
            NSLog(@"MGL MIP_DIAG texture glTex=%u target=0x%x size=%ux%u "
                  @"glLevels=%u mipmapLevels=%u mtlLevels=%lu base=%u max=%u "
                  @"mipmapped=%d genmipmaps=%d renderTarget=%d mtlTex=%p",
                  (unsigned)tex->name,
                  (unsigned)tex->target,
                  (unsigned)tex->width,
                  (unsigned)tex->height,
                  (unsigned)tex->num_levels,
                  (unsigned)tex->mipmap_levels,
                  (unsigned long)(mtlTex ? mtlTex.mipmapLevelCount : 0u),
                  (unsigned)tex->params.base_level,
                  (unsigned)tex->params.max_level,
                  tex->mipmapped ? 1 : 0,
                  tex->genmipmaps ? 1 : 0,
                  tex->is_render_target ? 1 : 0,
                  mtlTex);
        }
    }

    return true;
}

@end
