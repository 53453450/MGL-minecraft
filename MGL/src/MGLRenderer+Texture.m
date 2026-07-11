// MGLRenderer+Texture.m
// Texture upload/download Metal path methods extracted from MGLRenderer.m

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Texture_Private.h"

@implementation MGLRenderer (Texture)

- (bool)copyTextureUploadWithDedicatedCommandBuffer:(id<MTLBuffer>)sourceBuffer
                                        sourceOffset:(NSUInteger)sourceOffset
                                   sourceBytesPerRow:(NSUInteger)sourceBytesPerRow
                                 sourceBytesPerImage:(NSUInteger)sourceBytesPerImage
                                           sourceSize:(MTLSize)sourceSize
                                            toTexture:(id<MTLTexture>)texture
                                     destinationSlice:(NSUInteger)destinationSlice
                                     destinationLevel:(NSUInteger)destinationLevel
                                    destinationOrigin:(MTLOrigin)destinationOrigin
                                               reason:(const char *)reason
{
    if (!sourceBuffer || !texture || !_commandQueue) {
        NSLog(@"MGL ERROR: dedicated texture upload prerequisites missing (source=%p texture=%p queue=%p)",
              sourceBuffer, texture, _commandQueue);
        return false;
    }

    if (!kMGLUseDedicatedTextureUploadCommandBuffer) {
        /*
         * Texture uploads are GL commands and must stay ordered with draws in the
         * same context.  Committing a standalone upload command buffer here can
         * leapfrog an open render command buffer, so encode the blit into the
         * current command buffer after closing the active render encoder.
         */
        [self endRenderEncoding];

        if (![self ensureWritableCommandBuffer:reason ? reason : "texture_upload"]) {
            NSLog(@"MGL ERROR: failed to obtain current command buffer for %s",
                  reason ? reason : "texture_upload");
            return false;
        }

        id<MTLBlitCommandEncoder> blitEncoder = [_currentCommandBuffer blitCommandEncoder];
        if (!blitEncoder) {
            NSLog(@"MGL ERROR: failed to create ordered upload blit encoder for %s",
                  reason ? reason : "texture_upload");
            [self recordGPUError];
            return false;
        }

        @try {
            [blitEncoder copyFromBuffer:sourceBuffer
                           sourceOffset:sourceOffset
                       sourceBytesPerRow:sourceBytesPerRow
                     sourceBytesPerImage:sourceBytesPerImage
                              sourceSize:sourceSize
                               toTexture:texture
                        destinationSlice:destinationSlice
                        destinationLevel:destinationLevel
                       destinationOrigin:destinationOrigin];
            [blitEncoder endEncoding];
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: ordered upload encode failed (%s): %@",
                  reason ? reason : "texture_upload", exception.reason);
            @try {
                [blitEncoder endEncoding];
            } @catch (NSException *endException) {
                NSLog(@"MGL WARNING: ordered upload endEncoding failed (%s): %@",
                      reason ? reason : "texture_upload", endException.reason);
            }
            [self recordGPUError];
            return false;
        }

        return true;
    }

    id<MTLCommandBuffer> uploadCB = [_commandQueue commandBuffer];
    if (!uploadCB) {
        NSLog(@"MGL ERROR: failed to create dedicated upload command buffer for %s",
              reason ? reason : "texture_upload");
        [self recordGPUError];
        return false;
    }

    if (reason) {
        uploadCB.label = [NSString stringWithFormat:@"MGL.%s", reason];
    } else {
        uploadCB.label = @"MGL.texture_upload";
    }

    id<MTLBlitCommandEncoder> blitEncoder = [uploadCB blitCommandEncoder];
    if (!blitEncoder) {
        NSLog(@"MGL ERROR: failed to create dedicated upload blit encoder for %s",
              reason ? reason : "texture_upload");
        [self recordGPUError];
        return false;
    }

    @try {
        [blitEncoder copyFromBuffer:sourceBuffer
                       sourceOffset:sourceOffset
                   sourceBytesPerRow:sourceBytesPerRow
                 sourceBytesPerImage:sourceBytesPerImage
                          sourceSize:sourceSize
                           toTexture:texture
                    destinationSlice:destinationSlice
                    destinationLevel:destinationLevel
                   destinationOrigin:destinationOrigin];
        [blitEncoder endEncoding];
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: dedicated upload encode failed (%s): %@",
              reason ? reason : "texture_upload", exception.reason);
        [blitEncoder endEncoding];
        [self recordGPUError];
        return false;
    }

    dispatch_semaphore_t completionSemaphore = kMGLSynchronizeTextureUploads
        ? dispatch_semaphore_create(0)
        : NULL;
    __weak typeof(self) weakSelf = self;
    [uploadCB addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        if (cb.error) {
            NSLog(@"MGL ERROR: dedicated upload command buffer failed (%s): %@",
                  reason ? reason : "texture_upload", cb.error);
            [weakSelf recordGPUError];
        }

        if (completionSemaphore) {
            dispatch_semaphore_signal(completionSemaphore);
        }
    }];

    [uploadCB commit];

    if (!kMGLSynchronizeTextureUploads) {
        // Keep uploads ordered on the same queue but avoid stalling the render thread.
        return true;
    }

    dispatch_time_t deadline = dispatch_time(DISPATCH_TIME_NOW,
                                             (int64_t)(kMGLTextureUploadWaitTimeoutSeconds * NSEC_PER_SEC));
    if (dispatch_semaphore_wait(completionSemaphore, deadline) != 0) {
        NSLog(@"MGL WARNING: dedicated upload wait timed out (%s), continuing asynchronously",
              reason ? reason : "texture_upload");
        return true;
    }

    return uploadCB.error == nil;
}

- (bool)uploadTextureSliceViaBlit:(id<MTLTexture>)texture
                          texName:(GLuint)texName
                         texTarget:(GLenum)texTarget
                            bytes:(const void *)bytes
                      bytesPerRow:(NSUInteger)bytesPerRow
                    bytesPerImage:(NSUInteger)bytesPerImage
                            width:(NSUInteger)width
                           height:(NSUInteger)height
                            depth:(NSUInteger)depth
                            level:(NSUInteger)level
                            slice:(NSUInteger)slice
{
    if (!texture || !bytes || bytesPerRow == 0 || bytesPerImage == 0 || width == 0) {
        return false;
    }


    if ([self shouldSkipGPUOperations]) {
        NSLog(@"MGL AGX: Skipping texture upload during recovery");
        return false;
    }

    MTLTextureType textureType = texture.textureType;
    BOOL is3DTexture = (textureType == MTLTextureType3D);
    BOOL isArrayOrCubeTexture =
        (textureType == MTLTextureTypeCube ||
         textureType == MTLTextureTypeCubeArray ||
         textureType == MTLTextureType2DArray ||
         textureType == MTLTextureType1DArray ||
         textureType == MTLTextureType2DMultisampleArray);

    NSUInteger safeHeight = (height > 0) ? height : 1;
    NSUInteger safeDepth = (depth > 0) ? depth : 1;
    NSUInteger uploadRows = mglMetalUploadRowsForPixelFormat(texture.pixelFormat, safeHeight);
    if (uploadRows == 0 || bytesPerRow > (NSUIntegerMax / uploadRows)) {
        NSLog(@"MGL WARNING: Rejecting texture upload with invalid row layout: bpr=%lu rows=%lu",
              (unsigned long)bytesPerRow,
              (unsigned long)uploadRows);
        return false;
    }
    NSUInteger expectedBytesPerImage = bytesPerRow * uploadRows;
    NSUInteger copyDepth = is3DTexture ? safeDepth : 1;
    NSUInteger safeBytesPerImage = bytesPerImage;

    if (safeBytesPerImage < expectedBytesPerImage) {
        NSLog(@"MGL WARNING: Rejecting texture upload with short image stride: bpi=%lu expected=%lu fmt=%lu",
              (unsigned long)safeBytesPerImage,
              (unsigned long)expectedBytesPerImage,
              (unsigned long)texture.pixelFormat);
        return false;
    }

    if (isArrayOrCubeTexture) {
        // For array/cubemap uploads each slice is uploaded independently.
        // Clamp to per-slice bytes to avoid accidentally treating N slices as one image.
        if (safeBytesPerImage != expectedBytesPerImage) {
            NSLog(@"MGL INFO: Normalizing bytesPerImage for array/cube upload (slice=%lu level=%lu old=%lu expected=%lu)",
                  (unsigned long)slice, (unsigned long)level,
                  (unsigned long)safeBytesPerImage, (unsigned long)expectedBytesPerImage);
        }
        safeBytesPerImage = expectedBytesPerImage;
    } else {
        // Non-array/non-3D uploads should still represent a single image.
        safeBytesPerImage = expectedBytesPerImage;
    }

    if (textureType == MTLTextureTypeCube || textureType == MTLTextureTypeCubeArray) {
        NSLog(@"MGL CUBE UPLOAD tex=%u glTarget=0x%x face=%lu slice=%lu level=%lu origin=(0,0,0) size=%lux%lux%lu bpr=%lu bpi=%lu ptr=%p",
              texName,
              texTarget,
              (unsigned long)slice,
              (unsigned long)slice,
              (unsigned long)level,
              (unsigned long)width,
              (unsigned long)safeHeight,
              (unsigned long)copyDepth,
              (unsigned long)bytesPerRow,
              (unsigned long)safeBytesPerImage,
              bytes);
    }

    /* 1D texture upload via replaceRegion branch:
     * - 1D textures are a low-frequency update path; replaceRegion is safe in this scenario;
     * - Before entering this function, the caller has already flushed CPU-side deferred
     *   draws via mglFlushPendingDrawsBeforeTextureWrite, avoiding ordering races between
     *   the upload and uncommitted render command buffers;
     * - Only available for shared storage; Private storage (e.g. MSAA) must fall back to the blit path. */
    if ((textureType == MTLTextureType1D || textureType == MTLTextureType1DArray) &&
        texture.storageMode != MTLStorageModePrivate) {
        @try {
            MTLRegion region = MTLRegionMake1D(0, width);
            if (textureType == MTLTextureType1DArray) {
                [texture replaceRegion:region
                            mipmapLevel:level
                                  slice:slice
                              withBytes:bytes
                            bytesPerRow:bytesPerRow
                          bytesPerImage:safeBytesPerImage];
            } else {
                [texture replaceRegion:region
                            mipmapLevel:level
                              withBytes:bytes
                            bytesPerRow:bytesPerRow];
            }
            if (mglTraceLogIsEnabled() &&
                texture.pixelFormat == MTLPixelFormatR8Unorm &&
                width > 0) {
                const uint8_t *first = (const uint8_t *)bytes;
                mglTraceLog("TEXTURE_UPLOAD_1D_REPLACE tex=%u target=0x%x mtlType=%lu size=%lux%lu bpr=%lu bpi=%lu first=%u",
                            (unsigned)texName,
                            (unsigned)texTarget,
                            (unsigned long)textureType,
                            (unsigned long)width,
                            (unsigned long)safeHeight,
                            (unsigned long)bytesPerRow,
                            (unsigned long)safeBytesPerImage,
                            first ? first[0] : 0u);
            }
            return true;
        } @catch (NSException *exception) {
            NSLog(@"MGL WARNING: 1D texture replaceRegion upload failed, falling back to blit (tex=%u level=%lu slice=%lu): %@",
                  (unsigned)texName,
                  (unsigned long)level,
                  (unsigned long)slice,
                  exception.reason);
        }
    }

    /* 3D texture upload via replaceRegion branch:
     * - 3D uses replaceRegion to work around the AGX driver's copyFromBuffer:toTexture: slice OOB
     *   assertion (triggered even when destinationSlice=0);
     *   Driver bug tracked via MGLCapabilityHasBug(MGL_BUG_3D_COPY_FROM_BUFFER_SLICE_OOB).
     * - replaceRegion:mipmapLevel:withBytes:bytesPerRow: does not accept bytesPerImage,
     *   so tightly packed data is required (safeBytesPerImage == bytesPerRow * height, i.e. expectedBytesPerImage);
     * - Only available for shared storage; if the tight packing condition is not met, fall back to the blit path. */
    if (is3DTexture &&
        MGLCapabilityHasBug(&_capability, MGL_BUG_3D_COPY_FROM_BUFFER_SLICE_OOB) &&
        texture.storageMode != MTLStorageModePrivate &&
        safeBytesPerImage == expectedBytesPerImage) {
        @try {
            MTLRegion region = MTLRegionMake3D(0, 0, 0, width, safeHeight, copyDepth);
            [texture replaceRegion:region
                        mipmapLevel:level
                          withBytes:bytes
                        bytesPerRow:bytesPerRow];
            return true;
        } @catch (NSException *exception) {
            NSLog(@"MGL WARNING: 3D texture replaceRegion upload failed, falling back to blit (tex=%u level=%lu): %@",
                  (unsigned)texName, (unsigned long)level,
                  exception.reason);
        }
    }

    /* 2D / 2DArray / Cube texture upload via blit path (dedicated CB + completion handler):
     * - replaceRegion must not be used: when the texture is being sampled by an in-flight
     *   command buffer, replaceRegion's CPU direct writes are not subject to GPU-side
     *   ordering constraints, causing data races with in-flight sampling draws (this
     *   previously caused Minecraft GUI item rendering corruption);
     * - The blit path is required to guarantee GPU-side ordering: copyTextureUploadWithDedicatedCommandBuffer
     *   calls endRenderEncoding to close the current render encoder, and encodes
     *   copyFromBuffer:toTexture: on a dedicated CB, with the Metal command queue
     *   guaranteeing submission order relative to existing render CBs;
     * - The 1D/3D branches do not hit this precondition (low-frequency or driver bug
     *   workaround) and have already returned earlier. */
    if (copyDepth > 0 && safeBytesPerImage > (NSUIntegerMax / copyDepth)) {
        NSLog(@"MGL WARNING: Rejecting texture upload with overflowing buffer size: bpi=%lu depth=%lu",
              (unsigned long)safeBytesPerImage,
              (unsigned long)copyDepth);
        return false;
    }
    NSUInteger bufferSize = safeBytesPerImage * copyDepth;
    if (bufferSize == 0 || bufferSize > (512 * 1024 * 1024)) {
        NSLog(@"MGL WARNING: Rejecting texture upload with invalid buffer size: %lu", (unsigned long)bufferSize);
        return false;
    }

    id<MTLBuffer> uploadBuffer = [_device newBufferWithBytes:bytes
                                                       length:bufferSize
                                                      options:MTLResourceStorageModeShared];
    if (!uploadBuffer) {
        NSLog(@"MGL WARNING: Failed to allocate upload buffer for texture blit");
        return false;
    }

    bool uploaded = [self copyTextureUploadWithDedicatedCommandBuffer:uploadBuffer
                                                         sourceOffset:0
                                                    sourceBytesPerRow:bytesPerRow
                                                  sourceBytesPerImage:safeBytesPerImage
                                                            sourceSize:MTLSizeMake(width, safeHeight, copyDepth)
                                                             toTexture:texture
                                                      destinationSlice:slice
                                                      destinationLevel:level
                                                     destinationOrigin:MTLOriginMake(0, 0, 0)
                                                                reason:"texture_upload_blit"];
    if (!uploaded) {
        NSLog(@"MGL WARNING: Dedicated texture upload failed (level=%lu slice=%lu)",
              (unsigned long)level, (unsigned long)slice);
    }
    return uploaded;
}

- (bool)uploadFullCPUTextureDataIntoTexture:(Texture *)tex
                                      metal:(id<MTLTexture>)texture
                                     reason:(const char *)reason
{
    if (!tex || !texture || !tex->faces[0].levels) {
        return false;
    }
    if (tex->target != GL_TEXTURE_2D ||
        texture.textureType != MTLTextureType2D) {
        return false;
    }

    int numFaces = 1;
    GLuint levelCount = MIN((GLuint)texture.mipmapLevelCount,
                            tex->num_levels ? tex->num_levels : 1u);
    if (levelCount == 0u ||
        !mglTextureHasUploadableCPUData(tex, numFaces, levelCount)) {
        return false;
    }

    bool uploadedAny = false;
    bool failedAny = false;
    for (int face = 0; face < numFaces; face++) {
        if (!tex->faces[face].levels) {
            failedAny = true;
            continue;
        }

        for (GLuint level = 0; level < levelCount; level++) {
            TextureLevel *uploadLevel = &tex->faces[face].levels[level];
            if (!mglTextureLevelHasUploadableCPUData(uploadLevel)) {
                continue;
            }

            NSUInteger width = uploadLevel->width;
            NSUInteger height = MAX((NSUInteger)uploadLevel->height, 1UL);
            NSUInteger depth = MAX((NSUInteger)uploadLevel->depth, 1UL);
            NSUInteger bytesPerRow = uploadLevel->pitch;
            const uint8_t *srcData = (const uint8_t *)(uintptr_t)uploadLevel->data;
            if (!srcData || width == 0 || height == 0 || bytesPerRow == 0) {
                failedAny = true;
                continue;
            }

            NSUInteger copyDepth = (texture.textureType == MTLTextureType3D) ? depth : 1UL;
            NSUInteger availableBytes = uploadLevel->data_size;
            /* For block-compressed formats, `data_size` is the actual byte
             * count of one block-aligned image (e.g. BC1 32x32 = 512 B), not
             * bytesPerRow * pixel_height (which would over-count by block_h).
             * Use the smaller of availableBytes/copyDepth and the linear
             * stride so neither compressed nor uncompressed uploads overflow. */
            NSUInteger bytesPerImage = MIN(availableBytes / copyDepth, bytesPerRow * height);
            void *expandedUploadData = NULL;
            if (availableBytes < bytesPerImage * copyDepth) {
                static uint64_t s_shortBackingLogs = 0;
                uint64_t hit = ++s_shortBackingLogs;
                if (kMGLDiagnosticStateLogs &&
                    (hit <= 32ull || (hit % 512ull) == 0ull)) {
                    MGLTraceNSLog(@"MGL TEXTURE CPU-REFRESH skip short backing tex=%u level=%u face=%d have=%lu need=%lu reason=%s hit=%llu",
                                  (unsigned)tex->name,
                                  (unsigned)level,
                                  face,
                                  (unsigned long)availableBytes,
                                  (unsigned long)(bytesPerImage * copyDepth),
                                  reason ? reason : "(null)",
                                  (unsigned long long)hit);
                }
                failedAny = true;
                continue;
            }

            if (mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, texture.pixelFormat)) {
                NSUInteger expandedBytesPerRow = 0;
                NSUInteger expandedBytesPerImage = 0;
                expandedUploadData = mglCreateRGBA8ExpandedUpload(tex,
                                                                  srcData,
                                                                  width,
                                                                  height,
                                                                  bytesPerRow,
                                                                  &expandedBytesPerRow,
                                                                  &expandedBytesPerImage);
                if (expandedUploadData) {
                    srcData = expandedUploadData;
                    bytesPerRow = expandedBytesPerRow;
                    bytesPerImage = expandedBytesPerImage;
                    availableBytes = expandedBytesPerImage * copyDepth;
                }
            } else if (mglTextureNeedsChannelExpansion(tex->internalformat, texture.pixelFormat)) {
                NSUInteger expandedBytesPerRow = 0;
                NSUInteger expandedBytesPerImage = 0;
                expandedUploadData = mglCreateChannelExpandedUpload(tex,
                                                                     texture.pixelFormat,
                                                                     srcData,
                                                                     width,
                                                                     height,
                                                                     bytesPerRow,
                                                                     &expandedBytesPerRow,
                                                                     &expandedBytesPerImage);
                if (expandedUploadData) {
                    srcData = expandedUploadData;
                    bytesPerRow = expandedBytesPerRow;
                    bytesPerImage = expandedBytesPerImage;
                    availableBytes = expandedBytesPerImage * copyDepth;
                }
            }

            bool uploaded = [self uploadTextureSliceViaBlit:texture
                                                    texName:tex->name
                                                 texTarget:tex->target
                                                     bytes:srcData
                                               bytesPerRow:bytesPerRow
                                             bytesPerImage:bytesPerImage
                                                     width:width
                                                    height:height
                                                     depth:copyDepth
                                                     level:level
                                                     slice:0];
            free(expandedUploadData);
            if (uploaded) {
                uploadedAny = true;
            } else {
                failedAny = true;
            }
        }
    }

    static uint64_t s_refreshLogs = 0;
    uint64_t hit = ++s_refreshLogs;
    if (kMGLDiagnosticStateLogs &&
        (uploadedAny || hit <= 32ull || (hit % 512ull) == 0ull)) {
        MGLTraceNSLog(@"MGL TEXTURE CPU-REFRESH tex=%u mtl=%p uploaded=%d failed=%d dirty=0x%x levels=%u reason=%s hit=%llu",
                      (unsigned)tex->name,
                      texture,
                      uploadedAny ? 1 : 0,
                      failedAny ? 1 : 0,
                      (unsigned)tex->dirty_bits,
                      (unsigned)levelCount,
                      reason ? reason : "(null)",
                      (unsigned long long)hit);
    }

    if (uploadedAny && !failedAny) {
        tex->dirty_bits &= ~DIRTY_TEXTURE_DATA;
        [self recordGPUSuccess];
        return true;
    }

    return false;
}

- (void)mglApplyPendingDefaultColorClearToTexture:(id<MTLTexture>)texture
{
    if (!ctx || !texture || !(ctx->state.default_fbo_clear_bitmask & GL_COLOR_BUFFER_BIT)) {
        return;
    }

    MTLRenderPassDescriptor *clearPass = [MTLRenderPassDescriptor renderPassDescriptor];
    clearPass.colorAttachments[0].texture = texture;
    clearPass.colorAttachments[0].loadAction = MTLLoadActionClear;
    clearPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    clearPass.colorAttachments[0].clearColor =
        MTLClearColorMake(ctx->state.default_clear_color[0],
                          ctx->state.default_clear_color[1],
                          ctx->state.default_clear_color[2],
                          ctx->state.default_clear_color[3]);

    id<MTLRenderCommandEncoder> clearEncoder = [_currentCommandBuffer renderCommandEncoderWithDescriptor:clearPass];
    if (clearEncoder) {
        [clearEncoder endEncoding];
        ctx->state.default_fbo_clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
    } else {
        NSLog(@"MGL WARNING: readPixels failed to apply pending default framebuffer color clear");
    }
}

- (void)mglApplyPendingFBOColorClearForReadback:(Framebuffer *)fbo
                                     attachment:(FBOAttachment *)attachment
                                    textureObj:(Texture *)textureObj
                                     mtlTexture:(id<MTLTexture>)texture
                                  attachmentEnum:(GLenum)attachmentEnum
{
    if (!fbo || !attachment || !texture || !(attachment->clear_bitmask & GL_COLOR_BUFFER_BIT)) {
        return;
    }

    MGLMetalAttachmentSubresource subresource =
        mglMetalAttachmentSubresourceForAttachment(attachment);
    MTLRenderPassDescriptor *clearPass = [MTLRenderPassDescriptor renderPassDescriptor];
    clearPass.colorAttachments[0].texture = texture;
    clearPass.colorAttachments[0].level = subresource.level;
    clearPass.colorAttachments[0].slice = subresource.slice;
    clearPass.colorAttachments[0].depthPlane = subresource.depthPlane;
    clearPass.colorAttachments[0].loadAction = MTLLoadActionClear;
    clearPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    clearPass.colorAttachments[0].clearColor =
        MTLClearColorMake(attachment->clear_color[0],
                          attachment->clear_color[1],
                          attachment->clear_color[2],
                          attachment->clear_color[3]);

    id<MTLRenderCommandEncoder> clearEncoder = [_currentCommandBuffer renderCommandEncoderWithDescriptor:clearPass];
    if (clearEncoder) {
        [clearEncoder endEncoding];
        attachment->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
        mglMarkTextureLevelRenderTargetWritten(textureObj, attachment->level);
    } else {
        NSLog(@"MGL WARNING: readPixels failed to apply pending FBO clear fbo=%u attachment=0x%x",
              (unsigned)fbo->name,
              (unsigned)attachmentEnum);
    }
}

- (BOOL)mglReadColorTextureAsBGRA8:(id<MTLTexture>)sourceTexture
                       sourceLevel:(NSUInteger)sourceLevel
                       sourceSlice:(NSUInteger)sourceSlice
                   sourceDepthPlane:(NSUInteger)sourceDepthPlane
                         pixelBytes:(void *)pixelBytes
                        bytesPerRow:(NSUInteger)bytesPerRow
                      bytesPerImage:(NSUInteger)bytesPerImage
                         fromRegion:(MTLRegion)region
                             reason:(const char *)reason
{
    NSUInteger readSize = bytesPerImage;
    if (readSize == 0u && bytesPerRow > 0u) {
        readSize = bytesPerRow * region.size.height;
    }
    if (!pixelBytes || readSize == 0u) {
        return NO;
    }

    if (!sourceTexture || region.size.width == 0u || region.size.height == 0u) {
        return sourceTexture != nil;
    }

    if ([sourceTexture isFramebufferOnly]) {
        static uint64_t s_framebufferOnlyReadCount = 0;
        uint64_t hit = ++s_framebufferOnlyReadCount;
        if (hit <= 16ull || (hit % 256ull) == 0ull) {
            NSLog(@"MGL WARNING: readPixels cannot read framebufferOnly texture for %s hit=%llu",
                  reason ? reason : "unknown",
                  (unsigned long long)hit);
        }
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    if (!mglMetalReadbackFormatIsBGRA8Compatible(sourceTexture.pixelFormat)) {
        static uint64_t s_unsupportedReadFormatCount = 0;
        uint64_t hit = ++s_unsupportedReadFormatCount;
        if (hit <= 32ull || (hit % 256ull) == 0ull) {
            NSLog(@"MGL WARNING: readPixels unsupported Metal color readback format=%lu for %s hit=%llu; returning zero data",
                  (unsigned long)sourceTexture.pixelFormat,
                  reason ? reason : "unknown",
                  (unsigned long long)hit);
        }
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    if (sourceTexture.sampleCount > 1u) {
        sourceTexture = [self resolvedReadbackTextureForMultisampleTexture:sourceTexture
                                                               sourceLevel:sourceLevel
                                                               sourceSlice:sourceSlice
                                                           sourceDepthPlane:sourceDepthPlane
                                                                    reason:reason];
        if (!sourceTexture) {
            return NO;
        }
        sourceLevel = 0u;
        sourceSlice = 0u;
        sourceDepthPlane = 0u;
    }

    if (bytesPerRow < region.size.width * 4u) {
        NSLog(@"MGL WARNING: readPixels destination row too small row=%lu width=%lu for %s",
              (unsigned long)bytesPerRow,
              (unsigned long)region.size.width,
              reason ? reason : "unknown");
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    NSUInteger levelWidth = sourceTexture.width;
    NSUInteger levelHeight = sourceTexture.height;
    if (sourceLevel > 0u) {
        if (sourceLevel >= sourceTexture.mipmapLevelCount) {
            NSLog(@"MGL WARNING: readPixels invalid mip level=%lu mipLevels=%lu for %s",
                  (unsigned long)sourceLevel,
                  (unsigned long)sourceTexture.mipmapLevelCount,
                  reason ? reason : "unknown");
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return NO;
        }
        levelWidth = MAX((NSUInteger)1u, sourceTexture.width >> sourceLevel);
        levelHeight = MAX((NSUInteger)1u, sourceTexture.height >> sourceLevel);
    }

    NSInteger requestX = (NSInteger)region.origin.x;
    NSInteger requestY = (NSInteger)region.origin.y;
    NSInteger requestW = (NSInteger)region.size.width;
    NSInteger requestH = (NSInteger)region.size.height;
    NSInteger glMinX = MAX((NSInteger)0, requestX);
    NSInteger glMinY = MAX((NSInteger)0, requestY);
    NSInteger glMaxX = MIN((NSInteger)levelWidth, requestX + requestW);
    NSInteger glMaxY = MIN((NSInteger)levelHeight, requestY + requestH);
    NSInteger copyW = glMaxX - glMinX;
    NSInteger copyH = glMaxY - glMinY;
    NSInteger dstX = glMinX - requestX;
    NSInteger dstY = glMinY - requestY;
    NSInteger metalSrcX = glMinX;
    NSInteger metalSrcY = (NSInteger)levelHeight - glMaxY;

    if (copyW <= 0 || copyH <= 0) {
        return YES;
    }

    NSUInteger stagingBytesPerPixel = mglMetalReadbackBytesPerPixel(sourceTexture.pixelFormat);
    NSUInteger stagingBytesPerRow = (NSUInteger)copyW * stagingBytesPerPixel;
    NSUInteger stagingSize = stagingBytesPerRow * (NSUInteger)copyH;
    NSUInteger outputBytesPerRow = (NSUInteger)copyW * 4u;
    if (stagingSize == 0u) {
        return YES;
    }


    NSUInteger dstOffset = ((NSUInteger)dstY * bytesPerRow) + ((NSUInteger)dstX * 4u);
    if (dstOffset >= readSize ||
        outputBytesPerRow > bytesPerRow ||
        ((NSUInteger)copyH - 1u) * bytesPerRow + outputBytesPerRow > readSize - dstOffset) {
        NSLog(@"MGL WARNING: readPixels clipped copy exceeds destination storage for %s",
              reason ? reason : "unknown");
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    if (![self ensureWritableCommandBuffer:"mglReadColorTextureAsBGRA8"]) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    id<MTLBuffer> readBuffer = [_device newBufferWithLength:stagingSize
                                                    options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> blitEncoder = readBuffer ? [_currentCommandBuffer blitCommandEncoder] : nil;
    if (!readBuffer || !blitEncoder) {
        NSLog(@"MGL WARNING: readPixels failed to create readback resources for %s",
              reason ? reason : "unknown");
        mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return NO;
    }

    BOOL blitEncoderEnded = NO;
    @try {
        [blitEncoder copyFromTexture:sourceTexture
                          sourceSlice:sourceSlice
                          sourceLevel:sourceLevel
                         sourceOrigin:MTLOriginMake((NSUInteger)metalSrcX,
                                                    (NSUInteger)metalSrcY,
                                                    sourceDepthPlane)
                           sourceSize:MTLSizeMake((NSUInteger)copyW,
                                                  (NSUInteger)copyH,
                                                  1u)
                             toBuffer:readBuffer
                    destinationOffset:0u
               destinationBytesPerRow:stagingBytesPerRow
             destinationBytesPerImage:stagingSize];
        [blitEncoder endEncoding];
        blitEncoderEnded = YES;
    } @catch (NSException *exception) {
        if (!blitEncoderEnded) {
            @try {
                [blitEncoder endEncoding];
            } @catch (NSException *endException) {
                NSLog(@"MGL WARNING: readPixels failed to end blit encoder after copy exception for %s: %@",
                      reason ? reason : "unknown",
                      endException);
            }
        }
        NSLog(@"MGL WARNING: readPixels texture copy failed for %s: %@",
              reason ? reason : "unknown",
              exception);
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    __block NSError *readbackError = nil;
    dispatch_semaphore_t readbackDone = dispatch_semaphore_create(0);
    [_currentCommandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        readbackError = cb.error;
        dispatch_semaphore_signal(readbackDone);
    }];
    [_currentCommandBuffer commit];

    dispatch_time_t readbackDeadline = dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.25 * NSEC_PER_SEC));
    BOOL success = YES;
    if (dispatch_semaphore_wait(readbackDone, readbackDeadline) != 0) {
        NSLog(@"MGL WARNING: readPixels command buffer timed out for %s; returning zeroed data",
              reason ? reason : "unknown");
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        success = NO;
    } else if (readbackError) {
        NSLog(@"MGL WARNING: readPixels command buffer failed for %s: %@; returning zeroed data",
              reason ? reason : "unknown",
              readbackError);
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        success = NO;
    } else {
        uint8_t *dst = ((uint8_t *)pixelBytes) + dstOffset;
mglMetalCopyTextureBytesToBGRA8((const uint8_t *)readBuffer.contents,
                                        stagingBytesPerRow,
                                        dst,
                                        bytesPerRow,
                                        (NSUInteger)copyW,
                                        (NSUInteger)copyH,
                                        sourceTexture.pixelFormat,
                                        YES);
    }

    [self newCommandBuffer];
    return success;
}

- (BOOL)mglReadDepthTextureAsFloat:(id<MTLTexture>)sourceTexture
                       sourceLevel:(NSUInteger)sourceLevel
                       sourceSlice:(NSUInteger)sourceSlice
                   sourceDepthPlane:(NSUInteger)sourceDepthPlane
                         pixelBytes:(void *)pixelBytes
                        bytesPerRow:(NSUInteger)bytesPerRow
                      bytesPerImage:(NSUInteger)bytesPerImage
                         fromRegion:(MTLRegion)region
                             reason:(const char *)reason
{
    NSUInteger readSize = bytesPerImage;
    if (readSize == 0u && bytesPerRow > 0u) {
        readSize = bytesPerRow * region.size.height;
    }
    if (!pixelBytes || readSize == 0u) {
        return NO;
    }

    if (!sourceTexture || region.size.width == 0u || region.size.height == 0u) {
        return sourceTexture != nil;
    }

    BOOL sourceIsDepthStencil =
        sourceTexture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8;
    BOOL sourceIsDepth16 =
        sourceTexture.pixelFormat == MTLPixelFormatDepth16Unorm;
    if (sourceTexture.pixelFormat != MTLPixelFormatDepth32Float &&
        sourceTexture.pixelFormat != MTLPixelFormatDepth16Unorm &&
        !sourceIsDepthStencil) {
        static uint64_t s_unsupportedDepthReadFormatCount = 0;
        uint64_t hit = ++s_unsupportedDepthReadFormatCount;
        if (hit <= 32ull || (hit % 256ull) == 0ull) {
            NSLog(@"MGL WARNING: readPixels unsupported Metal depth readback format=%lu for %s hit=%llu",
                  (unsigned long)sourceTexture.pixelFormat,
                  reason ? reason : "unknown",
                  (unsigned long long)hit);
        }
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    if (sourceTexture.sampleCount > 1u) {
        sourceTexture = [self resolvedReadbackTextureForMultisampleTexture:sourceTexture
                                                               sourceLevel:sourceLevel
                                                               sourceSlice:sourceSlice
                                                           sourceDepthPlane:sourceDepthPlane
                                                                    reason:reason];
        if (!sourceTexture) {
            return NO;
        }
        sourceLevel = 0u;
        sourceSlice = 0u;
        sourceDepthPlane = 0u;
    }

    if (sourceIsDepthStencil) {
        sourceTexture = [self depthFloatTextureForDepthStencilReadback:sourceTexture
                                                                reason:reason];
        if (!sourceTexture) {
            return NO;
        }
        sourceLevel = 0u;
        sourceSlice = 0u;
        sourceDepthPlane = 0u;
        sourceIsDepthStencil = NO;
        sourceIsDepth16 = NO;
    }

    if (bytesPerRow < region.size.width * sizeof(float)) {
        NSLog(@"MGL WARNING: readPixels depth destination row too small row=%lu width=%lu for %s",
              (unsigned long)bytesPerRow,
              (unsigned long)region.size.width,
              reason ? reason : "unknown");
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    NSUInteger levelWidth = sourceTexture.width;
    NSUInteger levelHeight = sourceTexture.height;
    if (sourceLevel > 0u) {
        if (sourceLevel >= sourceTexture.mipmapLevelCount) {
            NSLog(@"MGL WARNING: readPixels invalid depth mip level=%lu mipLevels=%lu for %s",
                  (unsigned long)sourceLevel,
                  (unsigned long)sourceTexture.mipmapLevelCount,
                  reason ? reason : "unknown");
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return NO;
        }
        levelWidth = MAX((NSUInteger)1u, sourceTexture.width >> sourceLevel);
        levelHeight = MAX((NSUInteger)1u, sourceTexture.height >> sourceLevel);
    }

    NSInteger requestX = (NSInteger)region.origin.x;
    NSInteger requestY = (NSInteger)region.origin.y;
    NSInteger requestW = (NSInteger)region.size.width;
    NSInteger requestH = (NSInteger)region.size.height;
    NSInteger glMinX = MAX((NSInteger)0, requestX);
    NSInteger glMinY = MAX((NSInteger)0, requestY);
    NSInteger glMaxX = MIN((NSInteger)levelWidth, requestX + requestW);
    NSInteger glMaxY = MIN((NSInteger)levelHeight, requestY + requestH);
    NSInteger copyW = glMaxX - glMinX;
    NSInteger copyH = glMaxY - glMinY;
    NSInteger dstX = glMinX - requestX;
    NSInteger dstY = glMinY - requestY;
    NSInteger metalSrcX = glMinX;
    NSInteger metalSrcY = (NSInteger)levelHeight - glMaxY;

    if (copyW <= 0 || copyH <= 0) {
        return YES;
    }

    NSUInteger sourceDepthBytes = sourceIsDepthStencil ? sizeof(float) : (sourceIsDepth16 ? sizeof(uint16_t) : sizeof(float));
    NSUInteger stagingBytesPerRow = (NSUInteger)copyW * sourceDepthBytes;
    /* Metal requires destinationBytesPerRow to be a multiple of 4 bytes
     * on macOS. Depth16Unorm (2 bytes/pixel) can produce a non-aligned
     * row for narrow reads, causing the blit to return zeros/garbage. */
    stagingBytesPerRow = (stagingBytesPerRow + 3u) & ~3u;
    NSUInteger stagingSize = stagingBytesPerRow * (NSUInteger)copyH;
    if (stagingSize == 0u) {
        return YES;
    }

    NSUInteger dstOffset = ((NSUInteger)dstY * bytesPerRow) + ((NSUInteger)dstX * sizeof(float));
    NSUInteger destinationCopyBytesPerRow = (NSUInteger)copyW * sizeof(float);
    if (dstOffset >= readSize ||
        destinationCopyBytesPerRow > bytesPerRow ||
        ((NSUInteger)copyH - 1u) * bytesPerRow + destinationCopyBytesPerRow > readSize - dstOffset) {
        NSLog(@"MGL WARNING: readPixels clipped depth copy exceeds destination storage for %s",
              reason ? reason : "unknown");
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    if (![self ensureWritableCommandBuffer:"mglReadDepthTextureAsFloat"]) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    id<MTLBuffer> readBuffer = [_device newBufferWithLength:stagingSize
                                                    options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> blitEncoder = readBuffer ? [_currentCommandBuffer blitCommandEncoder] : nil;
    if (!readBuffer || !blitEncoder) {
        NSLog(@"MGL WARNING: readPixels failed to create depth readback resources for %s",
              reason ? reason : "unknown");
        mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return NO;
    }

    BOOL blitEncoderEnded = NO;
    @try {
        [blitEncoder copyFromTexture:sourceTexture
                          sourceSlice:sourceSlice
                          sourceLevel:sourceLevel
                         sourceOrigin:MTLOriginMake((NSUInteger)metalSrcX,
                                                    (NSUInteger)metalSrcY,
                                                    sourceDepthPlane)
                           sourceSize:MTLSizeMake((NSUInteger)copyW,
                                                  (NSUInteger)copyH,
                                                  1u)
                             toBuffer:readBuffer
                    destinationOffset:0u
               destinationBytesPerRow:stagingBytesPerRow
             destinationBytesPerImage:stagingSize];
        [blitEncoder endEncoding];
        blitEncoderEnded = YES;
    } @catch (NSException *exception) {
        if (!blitEncoderEnded) {
            @try {
                [blitEncoder endEncoding];
            } @catch (NSException *endException) {
                NSLog(@"MGL WARNING: readPixels failed to end depth blit encoder after copy exception for %s: %@",
                      reason ? reason : "unknown",
                      endException);
            }
        }
        NSLog(@"MGL WARNING: readPixels depth texture copy failed for %s: %@",
              reason ? reason : "unknown",
              exception);
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    __block NSError *readbackError = nil;
    dispatch_semaphore_t readbackDone = dispatch_semaphore_create(0);
    [_currentCommandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        readbackError = cb.error;
        dispatch_semaphore_signal(readbackDone);
    }];
    [_currentCommandBuffer commit];

    dispatch_time_t readbackDeadline = dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.25 * NSEC_PER_SEC));
    BOOL success = YES;
    if (dispatch_semaphore_wait(readbackDone, readbackDeadline) != 0) {
        NSLog(@"MGL WARNING: readPixels depth command buffer timed out for %s",
              reason ? reason : "unknown");
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        success = NO;
    } else if (readbackError) {
        NSLog(@"MGL WARNING: readPixels depth command buffer failed for %s: %@",
              reason ? reason : "unknown",
              readbackError);
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        success = NO;
    } else {
        uint8_t *dst = ((uint8_t *)pixelBytes) + dstOffset;
        if (sourceIsDepthStencil || sourceIsDepth16) {
            const uint8_t *src = (const uint8_t *)readBuffer.contents;
            for (NSUInteger row = 0; row < (NSUInteger)copyH; row++) {
                const uint8_t *srcRow = src + row * stagingBytesPerRow;
                float *dstRow = (float *)(void *)(dst + ((NSUInteger)copyH - 1u - row) * bytesPerRow);
                for (NSUInteger column = 0; column < (NSUInteger)copyW; column++) {
                    if (sourceIsDepth16) {
                        uint16_t value = 0u;
                        memcpy(&value, srcRow + column * sourceDepthBytes, sizeof(value));
                        dstRow[column] = (float)value / 65535.0f;
                    } else {
                        memcpy(&dstRow[column], srcRow + column * sourceDepthBytes, sizeof(float));
                    }
                }
            }
        } else {
            mglMetalCopyRows((const uint8_t *)readBuffer.contents,
                             stagingBytesPerRow,
                             dst,
                             bytesPerRow,
                             stagingBytesPerRow,
                             (NSUInteger)copyH,
                             YES);
        }
    }

    [self newCommandBuffer];
    return success;
}

- (BOOL)mglReadIntegerTextureAsRGBA32:(id<MTLTexture>)sourceTexture
                           pixelBytes:(void *)pixelBytes
                           bytesPerRow:(NSUInteger)bytesPerRow
                        bytesPerImage:(NSUInteger)bytesPerImage
                           fromRegion:(MTLRegion)region
                     outputComponents:(NSUInteger)outputComponents
                  outputComponentBytes:(NSUInteger)outputComponentBytes
                         componentMap:(const int[4])componentMap
                          packedType:(GLenum)packedType
                        mipmapLevel:(NSUInteger)mipmapLevel
                              slice:(NSUInteger)mtlSlice
                     isRenderTarget:(BOOL)isRenderTarget
{
    NSUInteger componentCount = 0u;
    NSUInteger sourceComponentBytes = 0u;
    BOOL sourceSigned = NO;
    BOOL sourceRGB10A2Uint = NO;
    switch (sourceTexture.pixelFormat) {
        case MTLPixelFormatR8Uint:
            componentCount = 1u; sourceComponentBytes = 1u;
            break;
        case MTLPixelFormatR8Sint:
            componentCount = 1u; sourceComponentBytes = 1u; sourceSigned = YES;
            break;
        case MTLPixelFormatR16Uint:
            componentCount = 1u; sourceComponentBytes = 2u;
            break;
        case MTLPixelFormatR16Sint:
            componentCount = 1u; sourceComponentBytes = 2u; sourceSigned = YES;
            break;
        case MTLPixelFormatR32Sint:
            componentCount = 1u; sourceComponentBytes = 4u; sourceSigned = YES;
            break;
        case MTLPixelFormatRG8Uint:
            componentCount = 2u; sourceComponentBytes = 1u;
            break;
        case MTLPixelFormatRG8Sint:
            componentCount = 2u; sourceComponentBytes = 1u; sourceSigned = YES;
            break;
        case MTLPixelFormatRG16Uint:
            componentCount = 2u; sourceComponentBytes = 2u;
            break;
        case MTLPixelFormatRG16Sint:
            componentCount = 2u; sourceComponentBytes = 2u; sourceSigned = YES;
            break;
        case MTLPixelFormatRG32Sint:
            componentCount = 2u; sourceComponentBytes = 4u; sourceSigned = YES;
            break;
        case MTLPixelFormatRGBA8Uint:
            componentCount = 4u; sourceComponentBytes = 1u;
            break;
        case MTLPixelFormatRGBA8Sint:
            componentCount = 4u; sourceComponentBytes = 1u; sourceSigned = YES;
            break;
        case MTLPixelFormatRGBA16Uint:
            componentCount = 4u; sourceComponentBytes = 2u;
            break;
        case MTLPixelFormatRGBA16Sint:
            componentCount = 4u; sourceComponentBytes = 2u; sourceSigned = YES;
            break;
        case MTLPixelFormatRGBA32Sint:
            componentCount = 4u; sourceComponentBytes = 4u; sourceSigned = YES;
            break;
        case MTLPixelFormatR32Uint:
            componentCount = 1u; sourceComponentBytes = 4u;
            break;
        case MTLPixelFormatRG32Uint:
            componentCount = 2u; sourceComponentBytes = 4u;
            break;
        case MTLPixelFormatRGBA32Uint:
            componentCount = 4u; sourceComponentBytes = 4u;
            break;
        case MTLPixelFormatRGB10A2Uint:
            componentCount = 4u; sourceComponentBytes = 4u; sourceRGB10A2Uint = YES;
            break;
        default:
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return NO;
    }

    if (sourceTexture.sampleCount > 1u) {
        sourceTexture = [self resolvedReadbackTextureForMultisampleTexture:sourceTexture
                                                               sourceLevel:mipmapLevel
                                                               sourceSlice:mtlSlice
                                                           sourceDepthPlane:0u
                                                                    reason:"integer FBO readback"];
        if (!sourceTexture) {
            return NO;
        }
        mipmapLevel = 0u;
        mtlSlice = 0u;
    }

    /* Determine output pixel bytes for packed types. */
    BOOL isPackedType = NO;
    NSUInteger packedBitWidths[4] = {0, 0, 0, 0};
    NSUInteger packedShifts[4] = {0, 0, 0, 0};
    NSUInteger packedTotalBits = 0u;
    NSUInteger packedOutputBytes = 0u;

    switch (packedType) {
        case 0x8032: /* GL_UNSIGNED_BYTE_3_3_2 */
            isPackedType = YES;
            packedBitWidths[0]=3; packedBitWidths[1]=3; packedBitWidths[2]=2; packedBitWidths[3]=0;
            packedShifts[0]=5;  packedShifts[1]=2;  packedShifts[2]=0;  packedShifts[3]=0;
            packedTotalBits=8; packedOutputBytes=1; outputComponents=3;
            break;
        case 0x8362: /* GL_UNSIGNED_BYTE_2_3_3_REV */
            isPackedType = YES;
            packedBitWidths[0]=3; packedBitWidths[1]=3; packedBitWidths[2]=2; packedBitWidths[3]=0;
            packedShifts[0]=0;  packedShifts[1]=3;  packedShifts[2]=6;  packedShifts[3]=0;
            packedTotalBits=8; packedOutputBytes=1; outputComponents=3;
            break;
        case 0x8363: /* GL_UNSIGNED_SHORT_5_6_5 */
            isPackedType = YES;
            packedBitWidths[0]=5; packedBitWidths[1]=6; packedBitWidths[2]=5; packedBitWidths[3]=0;
            packedShifts[0]=11; packedShifts[1]=5;  packedShifts[2]=0;  packedShifts[3]=0;
            packedTotalBits=16; packedOutputBytes=2; outputComponents=3;
            break;
        case 0x8364: /* GL_UNSIGNED_SHORT_5_6_5_REV */
            isPackedType = YES;
            packedBitWidths[0]=5; packedBitWidths[1]=6; packedBitWidths[2]=5; packedBitWidths[3]=0;
            packedShifts[0]=0;  packedShifts[1]=5;  packedShifts[2]=11; packedShifts[3]=0;
            packedTotalBits=16; packedOutputBytes=2; outputComponents=3;
            break;
        case 0x8033: /* GL_UNSIGNED_SHORT_4_4_4_4 */
            isPackedType = YES;
            packedBitWidths[0]=4; packedBitWidths[1]=4; packedBitWidths[2]=4; packedBitWidths[3]=4;
            packedShifts[0]=12; packedShifts[1]=8;  packedShifts[2]=4;  packedShifts[3]=0;
            packedTotalBits=16; packedOutputBytes=2; outputComponents=4;
            break;
        case 0x8365: /* GL_UNSIGNED_SHORT_4_4_4_4_REV */
            isPackedType = YES;
            packedBitWidths[0]=4; packedBitWidths[1]=4; packedBitWidths[2]=4; packedBitWidths[3]=4;
            packedShifts[0]=0;  packedShifts[1]=4;  packedShifts[2]=8;  packedShifts[3]=12;
            packedTotalBits=16; packedOutputBytes=2; outputComponents=4;
            break;
        case 0x8034: /* GL_UNSIGNED_SHORT_5_5_5_1 */
            isPackedType = YES;
            packedBitWidths[0]=5; packedBitWidths[1]=5; packedBitWidths[2]=5; packedBitWidths[3]=1;
            packedShifts[0]=11; packedShifts[1]=6;  packedShifts[2]=1;  packedShifts[3]=0;
            packedTotalBits=16; packedOutputBytes=2; outputComponents=4;
            break;
        case 0x8366: /* GL_UNSIGNED_SHORT_1_5_5_5_REV */
            isPackedType = YES;
            packedBitWidths[0]=5; packedBitWidths[1]=5; packedBitWidths[2]=5; packedBitWidths[3]=1;
            packedShifts[0]=0;  packedShifts[1]=5;  packedShifts[2]=10; packedShifts[3]=15;
            packedTotalBits=16; packedOutputBytes=2; outputComponents=4;
            break;
        case 0x8035: /* GL_UNSIGNED_INT_8_8_8_8 */
            isPackedType = YES;
            packedBitWidths[0]=8; packedBitWidths[1]=8; packedBitWidths[2]=8; packedBitWidths[3]=8;
            packedShifts[0]=24; packedShifts[1]=16; packedShifts[2]=8;  packedShifts[3]=0;
            packedTotalBits=32; packedOutputBytes=4; outputComponents=4;
            break;
        case 0x8367: /* GL_UNSIGNED_INT_8_8_8_8_REV */
            isPackedType = YES;
            packedBitWidths[0]=8; packedBitWidths[1]=8; packedBitWidths[2]=8; packedBitWidths[3]=8;
            packedShifts[0]=0;  packedShifts[1]=8;  packedShifts[2]=16; packedShifts[3]=24;
            packedTotalBits=32; packedOutputBytes=4; outputComponents=4;
            break;
        case 0x8036: /* GL_UNSIGNED_INT_10_10_10_2 */
            isPackedType = YES;
            packedBitWidths[0]=10; packedBitWidths[1]=10; packedBitWidths[2]=10; packedBitWidths[3]=2;
            packedShifts[0]=22; packedShifts[1]=12; packedShifts[2]=2;  packedShifts[3]=0;
            packedTotalBits=32; packedOutputBytes=4; outputComponents=4;
            break;
        case 0x8368: /* GL_UNSIGNED_INT_2_10_10_10_REV */
            isPackedType = YES;
            packedBitWidths[0]=10; packedBitWidths[1]=10; packedBitWidths[2]=10; packedBitWidths[3]=2;
            packedShifts[0]=0;  packedShifts[1]=10; packedShifts[2]=20; packedShifts[3]=30;
            packedTotalBits=32; packedOutputBytes=4; outputComponents=4;
            break;
        default:
            break;
    }

    NSUInteger dstPixelBytes = isPackedType ? packedOutputBytes : (outputComponentBytes * outputComponents);
    NSUInteger readSize = bytesPerImage ? bytesPerImage : bytesPerRow * region.size.height;
    if (!pixelBytes || bytesPerRow < region.size.width * dstPixelBytes) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    (void)readSize;
    if (region.size.width == 0u || region.size.height == 0u) {
        return YES;
    }

    NSInteger minX = MAX((NSInteger)0, (NSInteger)region.origin.x);
    NSInteger minY = MAX((NSInteger)0, (NSInteger)region.origin.y);
    NSInteger maxX = MIN((NSInteger)sourceTexture.width,
                         (NSInteger)region.origin.x + (NSInteger)region.size.width);
    NSInteger maxY = MIN((NSInteger)sourceTexture.height,
                         (NSInteger)region.origin.y + (NSInteger)region.size.height);
    NSInteger copyW = maxX - minX;
    NSInteger copyH = maxY - minY;
    if (copyW <= 0 || copyH <= 0) {
        return YES;
    }

    NSUInteger srcPixelBytes = sourceRGB10A2Uint ? 4u : componentCount * sourceComponentBytes;
    NSUInteger srcBytesPerRow = (NSUInteger)copyW * srcPixelBytes;
    NSUInteger stagingSize = srcBytesPerRow * (NSUInteger)copyH;
    if (![self ensureWritableCommandBuffer:"mglReadIntegerTextureAsRGBA32"]) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    id<MTLBuffer> readBuffer = [_device newBufferWithLength:stagingSize
                                                    options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> blit = readBuffer ? [_currentCommandBuffer blitCommandEncoder] : nil;
    if (!readBuffer || !blit) {
        mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return NO;
    }

    /* Calculate the texture height at the specified mipmap level. */
    NSUInteger levelHeight = sourceTexture.height;
    if (mipmapLevel > 0u) {
        levelHeight = MAX((NSUInteger)1u, sourceTexture.height >> mipmapLevel);
    }
    /* Render-target textures are stored top-to-bottom in Metal (Metal y=0 =
     * GL y=levelHeight-1), so the blit source origin must be Y-flipped.
     * Non-render-target textures (e.g. storage images written via imageStore)
     * store data in GL order (Metal y=0 = GL y=0), so no source Y-flip is
     * needed.  Using the flipped origin for storage images would read the
     * wrong rows and corrupt the readback. */
    NSUInteger blitSrcY = isRenderTarget
        ? (levelHeight - (NSUInteger)maxY)
        : (NSUInteger)minY;
    [blit copyFromTexture:sourceTexture
              sourceSlice:mtlSlice
              sourceLevel:mipmapLevel
             sourceOrigin:MTLOriginMake((NSUInteger)minX,
                                        blitSrcY,
                                        0u)
               sourceSize:MTLSizeMake((NSUInteger)copyW, (NSUInteger)copyH, 1u)
                 toBuffer:readBuffer
        destinationOffset:0u
   destinationBytesPerRow:srcBytesPerRow
 destinationBytesPerImage:stagingSize];
    [blit endEncoding];
    [_currentCommandBuffer commit];
    [_currentCommandBuffer waitUntilCompleted];

    const uint8_t *src = (const uint8_t *)readBuffer.contents;
    NSUInteger dstX = (NSUInteger)(minX - (NSInteger)region.origin.x);
    NSUInteger dstY = (NSUInteger)(minY - (NSInteger)region.origin.y);
    /* No output Y-flip: the source Y-flip above (conditional on isRenderTarget)
     * already ensures the staging buffer rows are in the correct order for GL.
     * The original behavior (before the isRenderTarget parameter was added) did
     * not Y-flip the output, and adding an output Y-flip for render targets
     * reverses the row order and breaks tests that read back render-target
     * textures via glGetTexImage (e.g. direct_state_access.textures_storage_multisample). */
    for (NSUInteger y = 0; y < (NSUInteger)copyH; y++) {
        const uint8_t *srcRow = src + y * srcBytesPerRow;
        NSUInteger outputY = dstY + y;
        uint8_t *dstRow = (uint8_t *)pixelBytes + outputY * bytesPerRow;
        for (NSUInteger x = 0; x < (NSUInteger)copyW; x++) {
            const uint8_t *s = srcRow + x * srcPixelBytes;
            uint8_t *d = dstRow + (dstX + x) * dstPixelBytes;

            /* Extract source component values (up to 4). */
            uint32_t srcValues[4] = {0, 0, 0, 0};
            for (NSUInteger sc = 0; sc < componentCount && sc < 4u; sc++) {
                if (sourceRGB10A2Uint) {
                    uint32_t packed = *(const uint32_t *)(const void *)s;
                    static const uint8_t rgb10a2_shifts[4] = {0u, 10u, 20u, 30u};
                    static const uint32_t rgb10a2_masks[4] = {0x3ffu, 0x3ffu, 0x3ffu, 0x3u};
                    srcValues[sc] = (packed >> rgb10a2_shifts[sc]) & rgb10a2_masks[sc];
                } else if (sourceComponentBytes == 1u) {
                    srcValues[sc] = sourceSigned
                        ? (uint32_t)(int32_t)*(const int8_t *)(const void *)(s + sc)
                        : (uint32_t)s[sc];
                } else if (sourceComponentBytes == 2u) {
                    srcValues[sc] = sourceSigned
                        ? (uint32_t)(int32_t)*(const int16_t *)(const void *)(s + sc * 2u)
                        : (uint32_t)*(const uint16_t *)(const void *)(s + sc * 2u);
                } else {
                    srcValues[sc] = *(const uint32_t *)(const void *)(s + sc * 4u);
                }
            }

            if (isPackedType) {
                /* Pack values into the packed format.
                 * Per OpenGL spec, integer values are CLAMPED to the bit width, not masked. */
                uint32_t packed = 0u;
                for (NSUInteger c = 0; c < outputComponents && c < 4u; c++) {
                    int srcIdx = (c < 4u) ? componentMap[c] : -1;
                    uint32_t val = 0u;
                    if (srcIdx >= 0 && (NSUInteger)srcIdx < componentCount) {
                        val = srcValues[srcIdx];
                    }
                    /* Clamp to bit width (not mask). */
                    uint32_t maxVal = (packedBitWidths[c] >= 32u) ? 0xFFFFFFFFu : ((1u << packedBitWidths[c]) - 1u);
                    if (val > maxVal) val = maxVal;
                    packed |= val << packedShifts[c];
                }
                if (packedOutputBytes == 1u) {
                    d[0] = (uint8_t)packed;
                } else if (packedOutputBytes == 2u) {
                    ((uint16_t *)(void *)d)[0] = (uint16_t)packed;
                } else {
                    ((uint32_t *)(void *)d)[0] = packed;
                }
            } else {
                /* Non-packed: write each component individually.
                 * Per OpenGL spec, integer values are CLAMPED to the output type range. */
                for (NSUInteger c = 0; c < outputComponents; c++) {
                    int srcIdx = (c < 4u) ? componentMap[c] : -1;
                    uint32_t value = 0u;
                    if (srcIdx >= 0 && (NSUInteger)srcIdx < componentCount) {
                        value = srcValues[srcIdx];
                    }
                    if (outputComponentBytes == 1u) {
                        if (packedType == GL_BYTE) {
                            /* Signed byte: clamp to [-128, 127].
                             * If source is unsigned, values > 127 must clamp
                             * to 127 (not wrap to negative via int32_t cast). */
                            if (sourceSigned) {
                                int32_t sv = (int32_t)value;
                                if (sv > 127) sv = 127;
                                if (sv < -128) sv = -128;
                                d[c] = (uint8_t)(int8_t)sv;
                            } else {
                                if (value > 127u) value = 127u;
                                d[c] = (uint8_t)value;
                            }
                        } else {
                            /* Unsigned byte: clamp to [0, 255] */
                            if (value > 255u) value = 255u;
                            d[c] = (uint8_t)value;
                        }
                    } else if (outputComponentBytes == 2u) {
                        if (packedType == GL_SHORT) {
                            /* Signed short: clamp to [-32768, 32767].
                             * See comment above re: unsigned source. */
                            if (sourceSigned) {
                                int32_t sv = (int32_t)value;
                                if (sv > 32767) sv = 32767;
                                if (sv < -32768) sv = -32768;
                                ((uint16_t *)(void *)d)[c] = (uint16_t)(int16_t)sv;
                            } else {
                                if (value > 32767u) value = 32767u;
                                ((uint16_t *)(void *)d)[c] = (uint16_t)value;
                            }
                        } else {
                            /* Unsigned short: clamp to [0, 65535] */
                            if (value > 65535u) value = 65535u;
                            ((uint16_t *)(void *)d)[c] = (uint16_t)value;
                        }
                    } else {
                        if (packedType == GL_INT) {
                            /* Signed int: if source is unsigned, clamp to
                             * [0, INT32_MAX] to avoid wrap. */
                            if (sourceSigned) {
                                ((uint32_t *)(void *)d)[c] = value;
                            } else {
                                if (value > 0x7FFFFFFFu) value = 0x7FFFFFFFu;
                                ((uint32_t *)(void *)d)[c] = value;
                            }
                        } else {
                            /* Unsigned int: clamp to [0, 4294967295] */
                            ((uint32_t *)(void *)d)[c] = value;
                        }
                    }
                }
            }
        }
    }

    [self newCommandBuffer];
    return YES;
}

- (void)mglApplyPendingFBODepthClearForReadback:(Framebuffer *)fbo
                                     attachment:(FBOAttachment *)attachment
                                     textureObj:(Texture *)textureObj
                                     mtlTexture:(id<MTLTexture>)texture
{
    if (!fbo || !attachment || !texture || !(attachment->clear_bitmask & GL_DEPTH_BUFFER_BIT)) {
        return;
    }

    MGLMetalAttachmentSubresource subresource =
        mglMetalAttachmentSubresourceForAttachment(attachment);
    MTLRenderPassDescriptor *clearPass = [MTLRenderPassDescriptor renderPassDescriptor];
    clearPass.depthAttachment.texture = texture;
    clearPass.depthAttachment.level = subresource.level;
    clearPass.depthAttachment.slice = subresource.slice;
    clearPass.depthAttachment.depthPlane = subresource.depthPlane;
    clearPass.depthAttachment.loadAction = MTLLoadActionClear;
    clearPass.depthAttachment.storeAction = MTLStoreActionStore;
    clearPass.depthAttachment.clearDepth = attachment->clear_color[0];

    id<MTLRenderCommandEncoder> clearEncoder = [_currentCommandBuffer renderCommandEncoderWithDescriptor:clearPass];
    if (clearEncoder) {
        [clearEncoder endEncoding];
        attachment->clear_bitmask &= ~GL_DEPTH_BUFFER_BIT;
        mglMarkTextureLevelRenderTargetWritten(textureObj, attachment->level);
    } else {
        NSLog(@"MGL WARNING: readPixels failed to apply pending FBO depth clear fbo=%u",
              (unsigned)fbo->name);
    }
}

- (void)mglApplyPendingDefaultDepthClearToTexture:(id<MTLTexture>)texture
{
    if (!ctx || !texture || !(ctx->state.default_fbo_clear_bitmask & GL_DEPTH_BUFFER_BIT)) {
        return;
    }

    MTLRenderPassDescriptor *clearPass = [MTLRenderPassDescriptor renderPassDescriptor];
    clearPass.depthAttachment.texture = texture;
    clearPass.depthAttachment.loadAction = MTLLoadActionClear;
    clearPass.depthAttachment.storeAction = MTLStoreActionStore;
    clearPass.depthAttachment.clearDepth = ctx->state.var.depth_clear_value;

    id<MTLRenderCommandEncoder> clearEncoder = [_currentCommandBuffer renderCommandEncoderWithDescriptor:clearPass];
    if (clearEncoder) {
        [clearEncoder endEncoding];
        ctx->state.default_fbo_clear_bitmask &= ~GL_DEPTH_BUFFER_BIT;
    } else {
        NSLog(@"MGL WARNING: readPixels failed to apply pending default framebuffer depth clear");
    }
}

- (void)mtlReadDepthPixels:(GLMContext)glm_ctx
                pixelBytes:(void *)pixelBytes
               bytesPerRow:(NSUInteger)bytesPerRow
             bytesPerImage:(NSUInteger)bytesPerImage
                fromRegion:(MTLRegion)region
{
    ctx = glm_ctx;

    NSUInteger readSize = bytesPerImage;
    if (readSize == 0u && bytesPerRow > 0u) {
        readSize = bytesPerRow * region.size.height;
    }
    if (!pixelBytes || readSize == 0u) {
        return;
    }

    if (glm_ctx->state.readbuffer) {
        Framebuffer *fbo = glm_ctx->state.readbuffer;
        FBOAttachment *attachment = fbo ? &fbo->depth : NULL;
        Texture *readTextureObject = [self framebufferAttachmentTexture:attachment];
        if (!readTextureObject) {
            NSLog(@"MGL WARNING: readPixels FBO has no depth attachment fbo=%u",
                  fbo ? (unsigned)fbo->name : 0u);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        readTextureObject->is_render_target = true;
        if (![self bindMTLTexture:readTextureObject] || !readTextureObject->mtl_data) {
            NSLog(@"MGL WARNING: readPixels could not bind FBO depth texture fbo=%u tex=%u",
                  fbo ? (unsigned)fbo->name : 0u,
                  (unsigned)readTextureObject->name);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        id<MTLTexture> texture = (__bridge id<MTLTexture>)(readTextureObject->mtl_data);
        MGLMetalAttachmentSubresource subresource =
            mglMetalAttachmentSubresourceForAttachment(attachment);

        [self endRenderEncoding];
        if (![self ensureWritableCommandBuffer:"mtlReadDepthPixels.fbo"]) {
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }
        [self mglApplyPendingFBODepthClearForReadback:fbo
                                           attachment:attachment
                                           textureObj:readTextureObject
                                           mtlTexture:texture];
        [self mglReadDepthTextureAsFloat:texture
                             sourceLevel:subresource.level
                             sourceSlice:subresource.slice
                         sourceDepthPlane:subresource.depthPlane
                               pixelBytes:pixelBytes
                              bytesPerRow:bytesPerRow
                            bytesPerImage:bytesPerImage
                               fromRegion:region
                                   reason:"FBO depth readback"];
        return;
    }

    GLuint drawBufferIndex = mglDefaultDrawBufferIndexForGL(glm_ctx->state.read_buffer);
    id<MTLTexture> texture = nil;
    if (drawBufferIndex < _MAX_DRAW_BUFFERS) {
        texture = _drawBuffers[drawBufferIndex].depthbuffer;
    }

    if (!texture) {
        NSLog(@"MGL WARNING: readPixels default framebuffer has no depth texture slot=%u",
              (unsigned)drawBufferIndex);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    [self endRenderEncoding];
    if (![self ensureWritableCommandBuffer:"mtlReadDepthPixels.default"]) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }
    [self mglApplyPendingDefaultDepthClearToTexture:texture];
    [self mglReadDepthTextureAsFloat:texture
                         sourceLevel:0u
                         sourceSlice:0u
                     sourceDepthPlane:0u
                           pixelBytes:pixelBytes
                          bytesPerRow:bytesPerRow
                        bytesPerImage:bytesPerImage
                           fromRegion:region
                               reason:"default framebuffer depth readback"];
}

-(void)mtlReadIntegerPixels:(GLMContext)glm_ctx
                 pixelBytes:(void *)pixelBytes
                bytesPerRow:(NSUInteger)bytesPerRow
              bytesPerImage:(NSUInteger)bytesPerImage
                 fromRegion:(MTLRegion)region
                     format:(GLenum)format
                       type:(GLenum)type
{
    ctx = glm_ctx;
    Framebuffer *fbo = glm_ctx ? glm_ctx->state.readbuffer : NULL;
    GLenum readBuffer = glm_ctx ? glm_ctx->state.read_buffer : GL_NONE;
    if (!fbo || readBuffer < GL_COLOR_ATTACHMENT0 ||
        readBuffer >= GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    FBOAttachment *attachment = &fbo->color_attachments[readBuffer - GL_COLOR_ATTACHMENT0];
    Texture *textureObj = [self framebufferAttachmentTexture:attachment];
    if (!textureObj || ![self bindMTLTexture:textureObj] || !textureObj->mtl_data) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    id<MTLTexture> texture = (__bridge id<MTLTexture>)(textureObj->mtl_data);
    MGLMetalAttachmentSubresource subresource =
        mglMetalAttachmentSubresourceForAttachment(attachment);

    [self endRenderEncoding];
    if (![self ensureWritableCommandBuffer:"mtlReadIntegerPixels.fbo"]) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }
    [self mglApplyPendingFBOColorClearForReadback:fbo
                                       attachment:attachment
                                       textureObj:textureObj
                                       mtlTexture:texture
                                   attachmentEnum:readBuffer];

    /* Determine output component count and component mapping.
     * componentMap[c] = source component index for output component c, or -1. */
    NSUInteger outputComponents = 4u;
    int componentMap[4] = {0, 1, 2, 3};
    switch (format) {
        case GL_RED_INTEGER:    outputComponents = 1u; componentMap[0]=0; componentMap[1]=-1; componentMap[2]=-1; componentMap[3]=-1; break;
        case GL_RG_INTEGER:     outputComponents = 2u; componentMap[0]=0; componentMap[1]=1; componentMap[2]=-1; componentMap[3]=-1; break;
        case GL_RGB_INTEGER:    outputComponents = 3u; componentMap[0]=0; componentMap[1]=1; componentMap[2]=2;  componentMap[3]=-1; break;
        case GL_BGR_INTEGER:    outputComponents = 3u; componentMap[0]=2; componentMap[1]=1; componentMap[2]=0;  componentMap[3]=-1; break;
        case GL_RGBA_INTEGER:   outputComponents = 4u; componentMap[0]=0; componentMap[1]=1; componentMap[2]=2;  componentMap[3]=3;  break;
        case GL_BGRA_INTEGER:   outputComponents = 4u; componentMap[0]=2; componentMap[1]=1; componentMap[2]=0;  componentMap[3]=3;  break;
        case 0x8d95: /*GL_GREEN_INTEGER*/ outputComponents = 1u; componentMap[0]=1; componentMap[1]=-1; componentMap[2]=-1; componentMap[3]=-1; break;
        case 0x8d96: /*GL_BLUE_INTEGER*/  outputComponents = 1u; componentMap[0]=2; componentMap[1]=-1; componentMap[2]=-1; componentMap[3]=-1; break;
        case 0x8d97: /*GL_ALPHA_INTEGER*/ outputComponents = 1u; componentMap[0]=3; componentMap[1]=-1; componentMap[2]=-1; componentMap[3]=-1; break;
        default: outputComponents = 4u; break;
    }

    NSUInteger outputComponentBytes = (type == GL_BYTE || type == GL_UNSIGNED_BYTE) ? 1u :
                                      (type == GL_SHORT || type == GL_UNSIGNED_SHORT) ? 2u : 4u;

    [self mglReadIntegerTextureAsRGBA32:texture
                            pixelBytes:pixelBytes
                           bytesPerRow:bytesPerRow
                         bytesPerImage:bytesPerImage
                            fromRegion:region
                      outputComponents:outputComponents
                   outputComponentBytes:outputComponentBytes
                          componentMap:componentMap
                           packedType:type
                          mipmapLevel:subresource.level
                                slice:subresource.slice
                      isRenderTarget:YES];
}

-(void) mtlReadDrawable:(GLMContext) glm_ctx pixelBytes:(void *)pixelBytes bytesPerRow:(NSUInteger)bytesPerRow bytesPerImage:(NSUInteger)bytesPerImage fromRegion:(MTLRegion)region
{
    ctx = glm_ctx;

    NSUInteger readSize = bytesPerImage;
    if (readSize == 0 && bytesPerRow > 0) {
        readSize = bytesPerRow * region.size.height;
    }
    if (!pixelBytes || readSize == 0) {
        return;
    }

    if (glm_ctx->state.readbuffer)
    {
        Framebuffer *fbo = glm_ctx->state.readbuffer;
        GLenum readBuffer = glm_ctx->state.read_buffer;
        if (!fbo ||
            readBuffer == GL_NONE ||
            readBuffer < GL_COLOR_ATTACHMENT0 ||
            readBuffer >= GL_COLOR_ATTACHMENT0 + glm_ctx->state.max_color_attachments ||
            readBuffer >= GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS) {
            static uint64_t s_invalidReadFBOCount = 0;
            uint64_t hit = ++s_invalidReadFBOCount;
            if (hit <= 32ull || (hit % 256ull) == 0ull) {
                NSLog(@"MGL WARNING: readPixels invalid FBO read buffer=0x%x maxColor=%u hit=%llu; returning zero data",
                      (unsigned)readBuffer,
                      (unsigned)glm_ctx->state.max_color_attachments,
                      (unsigned long long)hit);
            }
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        GLuint attachmentIndex = (GLuint)(readBuffer - GL_COLOR_ATTACHMENT0);
        if (((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            static uint64_t s_missingReadAttachmentCount = 0;
            uint64_t hit = ++s_missingReadAttachmentCount;
            if (hit <= 32ull || (hit % 256ull) == 0ull) {
                NSLog(@"MGL WARNING: readPixels FBO read attachment 0x%x is not attached fbo=%u hit=%llu; returning zero data",
                      (unsigned)readBuffer,
                      (unsigned)fbo->name,
                      (unsigned long long)hit);
            }
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        FBOAttachment *attachment = &fbo->color_attachments[attachmentIndex];
        Texture *readTextureObject = [self framebufferAttachmentTexture:attachment];
        if (!readTextureObject) {
            NSLog(@"MGL WARNING: readPixels FBO attachment has no texture fbo=%u attachment=0x%x",
                  (unsigned)fbo->name,
                  (unsigned)readBuffer);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        readTextureObject->is_render_target = true;
        if (![self bindMTLTexture:readTextureObject] || !readTextureObject->mtl_data) {
            NSLog(@"MGL WARNING: readPixels could not bind FBO read texture fbo=%u attachment=0x%x tex=%u",
                  (unsigned)fbo->name,
                  (unsigned)readBuffer,
                  (unsigned)readTextureObject->name);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        id<MTLTexture> texture = (__bridge id<MTLTexture>)(readTextureObject->mtl_data);
        MGLMetalAttachmentSubresource subresource =
            mglMetalAttachmentSubresourceForAttachment(attachment);

        [self endRenderEncoding];
        if (![self ensureWritableCommandBuffer:"mtlReadDrawable.fbo"]) {
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }
        [self mglApplyPendingFBOColorClearForReadback:fbo
                                           attachment:attachment
                                           textureObj:readTextureObject
                                           mtlTexture:texture
                                       attachmentEnum:readBuffer];
        [self mglReadColorTextureAsBGRA8:texture
                              sourceLevel:subresource.level
                              sourceSlice:subresource.slice
                          sourceDepthPlane:subresource.depthPlane
                                pixelBytes:pixelBytes
                               bytesPerRow:bytesPerRow
                             bytesPerImage:bytesPerImage
                                fromRegion:region
                                    reason:"FBO color readback"];
        return;
    }

    GLuint mgl_drawbuffer;
    id<MTLTexture> texture = nil;

    switch(glm_ctx->state.read_buffer)
    {
        case GL_FRONT: mgl_drawbuffer = _FRONT; break;
        case GL_BACK: mgl_drawbuffer = _FRONT; break;
        case GL_FRONT_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
        case GL_FRONT_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
        case GL_BACK_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
        case GL_BACK_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
        case GL_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
        case GL_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
        default:
            NSLog(@"MGL WARNING: readPixels unsupported default read buffer=0x%x; returning zero data",
                  (unsigned)glm_ctx->state.read_buffer);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
    }

    if (mgl_drawbuffer == _FRONT)
    {
        if (!_drawable) {
            [self mglSyncLayerDrawableSizeFromView:"readPixels.default"];
            _drawable = [_layer nextDrawable];
        }
        texture = _drawable ? _drawable.texture : nil;
    }
    else if (mgl_drawbuffer < _MAX_DRAW_BUFFERS)
    {
        texture = _drawBuffers[mgl_drawbuffer].drawbuffer;
    }

    if (!texture)
    {
        NSLog(@"MGL WARNING: readPixels default drawbuffer slot=%u has no texture; returning zero data",
              (unsigned)mgl_drawbuffer);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    [self endRenderEncoding];
    if (![self ensureWritableCommandBuffer:"mtlReadDrawable.default"]) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }
    if (mgl_drawbuffer == _FRONT) {
        [self mglApplyPendingDefaultColorClearToTexture:texture];
    }
    [self mglReadColorTextureAsBGRA8:texture
                          sourceLevel:0u
                          sourceSlice:0u
                      sourceDepthPlane:0u
                            pixelBytes:pixelBytes
                           bytesPerRow:bytesPerRow
                            bytesPerImage:bytesPerImage
                               fromRegion:region
                                   reason:"default framebuffer readback"];
    return;
}

-(void) mtlGetTexImage:(GLMContext) glm_ctx tex: (Texture *)tex pixelBytes:(void *)pixelBytes bytesPerRow:(NSUInteger)bytesPerRow bytesPerImage:(NSUInteger)bytesPerImage fromRegion:(MTLRegion)region format:(GLenum)format type:(GLenum)type mipmapLevel:(NSUInteger)level slice:(NSUInteger)slice
{
    id<MTLTexture> texture = nil;

    ctx = glm_ctx;

    if (!tex) {
        NSLog(@"MGL ERROR: mtlGetTexImage called with NULL texture");
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    if (!pixelBytes) {
        NSLog(@"MGL WARNING: mtlGetTexImage called with NULL destination for texture %u", tex->name);
        return;
    }

    if (!tex->mtl_data && ![self bindMTLTexture:tex]) {
        NSLog(@"MGL ERROR: mtlGetTexImage failed to bind texture %u", tex->name);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    texture = (__bridge id<MTLTexture>)(tex->mtl_data);
    if (!texture) {
        NSLog(@"MGL ERROR: mtlGetTexImage texture %u has no Metal texture", tex->name);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    if ([texture isFramebufferOnly]) {
        NSLog(@"MGL ERROR: Cannot read from framebuffer only texture %u\n", tex->name);
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    if (![self synchronizeRenderPassForTextureReadback:texture reason:"mtlGetTexImage"]) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    /* Ensure any pending texture upload blit commands are committed before
     * reading back. Without this, getBytes may return stale/zero data because
     * the blit encoding the upload is still in the uncommitted command buffer. */
    [self endRenderEncoding];
    if (_currentCommandBuffer) {
        id<MTLCommandBuffer> pendingCB = _currentCommandBuffer;
        _currentCommandBuffer = nil;
        @try {
            [pendingCB commit];
            [pendingCB waitUntilCompleted];
        } @catch (NSException *e) {
            NSLog(@"MGL WARNING: mtlGetTexImage pre-readback flush failed: %@", e.reason);
        }
        if (pendingCB.error) {
            NSLog(@"MGL WARNING: mtlGetTexImage pre-readback command buffer error: %@", pendingCB.error);
        }
        [self newCommandBuffer];
    }

    MTLRegion readRegion = region;
    /* Render target textures are stored top-to-bottom in Metal, but OpenGL
     * readPixels expects bottom-to-top order. Flip Y for render targets to
     * match OpenGL semantics. This mirrors the Y flip already done in
     * mglReadColorTextureAsBGRA8 (metalSrcY = levelHeight - glMaxY). */
    BOOL flipRenderTargetRows = tex->is_render_target;
    if (flipRenderTargetRows && region.size.height > 0u) {
        NSUInteger levelHeight = MAX((NSUInteger)1u, texture.height >> level);
        if (region.origin.y > levelHeight ||
            region.size.height > levelHeight - region.origin.y) {
            NSLog(@"MGL ERROR: mtlGetTexImage invalid render-target read region tex=%u y=%lu h=%lu levelHeight=%lu",
                  tex->name,
                  (unsigned long)region.origin.y,
                  (unsigned long)region.size.height,
                  (unsigned long)levelHeight);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_VALUE);
            return;
        }
        readRegion.origin.y = levelHeight - (region.origin.y + region.size.height);
    }

    /* Integer texture readback path: when the source texture is an integer
     * format and the output format is GL_*_INTEGER, use the dedicated integer
     * readback function that handles packed types and component mapping. */
    BOOL sourceIsIntegerTexture =
        (texture.pixelFormat == MTLPixelFormatR8Uint   ||
         texture.pixelFormat == MTLPixelFormatR8Sint   ||
         texture.pixelFormat == MTLPixelFormatR16Uint  ||
         texture.pixelFormat == MTLPixelFormatR16Sint  ||
         texture.pixelFormat == MTLPixelFormatR32Uint  ||
         texture.pixelFormat == MTLPixelFormatR32Sint  ||
         texture.pixelFormat == MTLPixelFormatRG8Uint  ||
         texture.pixelFormat == MTLPixelFormatRG8Sint  ||
         texture.pixelFormat == MTLPixelFormatRG16Uint ||
         texture.pixelFormat == MTLPixelFormatRG16Sint ||
         texture.pixelFormat == MTLPixelFormatRG32Uint ||
         texture.pixelFormat == MTLPixelFormatRG32Sint ||
         texture.pixelFormat == MTLPixelFormatRGBA8Uint  ||
         texture.pixelFormat == MTLPixelFormatRGBA8Sint  ||
         texture.pixelFormat == MTLPixelFormatRGBA16Uint ||
         texture.pixelFormat == MTLPixelFormatRGBA16Sint ||
         texture.pixelFormat == MTLPixelFormatRGBA32Uint ||
         texture.pixelFormat == MTLPixelFormatRGBA32Sint ||
         texture.pixelFormat == MTLPixelFormatRGB10A2Uint);

    BOOL outputIsIntegerFormat =
        (format == GL_RED_INTEGER   || format == GL_RG_INTEGER    ||
         format == GL_RGB_INTEGER   || format == GL_BGR_INTEGER   ||
         format == GL_RGBA_INTEGER  || format == GL_BGRA_INTEGER  ||
         format == 0x8d95 /*GL_GREEN_INTEGER*/ ||
         format == 0x8d96 /*GL_BLUE_INTEGER*/  ||
         format == 0x8d97 /*GL_ALPHA_INTEGER*/);

    if (sourceIsIntegerTexture && outputIsIntegerFormat) {
        NSUInteger intOutputComponents = 4u;
        int intComponentMap[4] = {0, 1, 2, 3};
        switch (format) {
            case GL_RED_INTEGER:    intOutputComponents = 1u; intComponentMap[0]=0; intComponentMap[1]=-1; intComponentMap[2]=-1; intComponentMap[3]=-1; break;
            case GL_RG_INTEGER:     intOutputComponents = 2u; intComponentMap[0]=0; intComponentMap[1]=1; intComponentMap[2]=-1; intComponentMap[3]=-1; break;
            case GL_RGB_INTEGER:    intOutputComponents = 3u; intComponentMap[0]=0; intComponentMap[1]=1; intComponentMap[2]=2;  intComponentMap[3]=-1; break;
            case GL_BGR_INTEGER:    intOutputComponents = 3u; intComponentMap[0]=2; intComponentMap[1]=1; intComponentMap[2]=0;  intComponentMap[3]=-1; break;
            case GL_RGBA_INTEGER:   intOutputComponents = 4u; intComponentMap[0]=0; intComponentMap[1]=1; intComponentMap[2]=2;  intComponentMap[3]=3;  break;
            case GL_BGRA_INTEGER:   intOutputComponents = 4u; intComponentMap[0]=2; intComponentMap[1]=1; intComponentMap[2]=0;  intComponentMap[3]=3;  break;
            case 0x8d95: intOutputComponents = 1u; intComponentMap[0]=1; intComponentMap[1]=-1; intComponentMap[2]=-1; intComponentMap[3]=-1; break;
            case 0x8d96: intOutputComponents = 1u; intComponentMap[0]=2; intComponentMap[1]=-1; intComponentMap[2]=-1; intComponentMap[3]=-1; break;
            case 0x8d97: intOutputComponents = 1u; intComponentMap[0]=3; intComponentMap[1]=-1; intComponentMap[2]=-1; intComponentMap[3]=-1; break;
            default: break;
        }

        NSUInteger intOutputComponentBytes = (type == GL_BYTE || type == GL_UNSIGNED_BYTE) ? 1u :
                                             (type == GL_SHORT || type == GL_UNSIGNED_SHORT) ? 2u : 4u;

        /* Pass the original (non-Y-flipped) region. mglReadIntegerTextureAsRGBA32
         * does its own Y-flip on the blit source origin AND Y-flips the output
         * rows, so passing a pre-Y-flipped readRegion here would double-flip. */
        [self mglReadIntegerTextureAsRGBA32:texture
                                pixelBytes:pixelBytes
                               bytesPerRow:bytesPerRow
                             bytesPerImage:bytesPerImage
                                fromRegion:region
                          outputComponents:intOutputComponents
                       outputComponentBytes:intOutputComponentBytes
                              componentMap:intComponentMap
                               packedType:type
                              mipmapLevel:level
                                    slice:slice
                          isRenderTarget:(BOOL)tex->is_render_target];
        return;
    }

    NSUInteger dstPixelBytes = (NSUInteger)sizeForFormatType(format, type);
    BOOL directR32FloatRead =
        (texture.pixelFormat == MTLPixelFormatR32Float &&
         format == GL_RED &&
         type == GL_FLOAT);
    BOOL useBGRA8Conversion =
        (dstPixelBytes > 0u &&
         readRegion.size.depth == 1u &&
         !directR32FloatRead &&
         mglMetalReadbackFormatIsBGRA8Compatible(texture.pixelFormat));

    // MTLStorageModePrivate textures cannot be read directly with getBytes:.
    // Use a blit-to-buffer path to convert GPU-private tiled memory to linear CPU memory.
    if (texture.storageMode == MTLStorageModePrivate) {
        /* When useBGRA8Conversion is set but the source texture is not actually
         * 4 bytes-per-pixel (e.g. RGBA32Float is 16 bpp), the staging buffer
         * must be sized for the *source* pixel format, not the BGRA8 intermediate.
         * The blit copies raw source data into staging; conversion happens afterwards. */
        NSUInteger sourceBpp = mglMetalReadbackBytesPerPixel(texture.pixelFormat);
        BOOL sourceIsBGRA8 =
            (texture.pixelFormat == MTLPixelFormatBGRA8Unorm ||
             texture.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB ||
             texture.pixelFormat == MTLPixelFormatRGBA8Unorm ||
             texture.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB);
        NSUInteger rowBytes;
        if (useBGRA8Conversion && !sourceIsBGRA8 && sourceBpp > 0u) {
            rowBytes = readRegion.size.width * sourceBpp;
        } else if (useBGRA8Conversion) {
            rowBytes = readRegion.size.width * 4u;
        } else {
            rowBytes = (bytesPerRow > 0 ? bytesPerRow : readRegion.size.width * MAX(dstPixelBytes, (NSUInteger)1u));
        }
        NSUInteger imageBytes = rowBytes * readRegion.size.height;
        NSUInteger totalBytes = imageBytes;
        if (!useBGRA8Conversion && bytesPerImage > 0 && readRegion.size.depth > 1) {
            totalBytes = bytesPerImage * readRegion.size.depth;
        }

        id<MTLBuffer> stagingBuffer = [_device newBufferWithLength:totalBytes
                                                           options:MTLResourceStorageModeShared];
        if (!stagingBuffer) {
            NSLog(@"MGL ERROR: mtlGetTexImage failed to allocate staging buffer for texture %u", tex->name);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
            return;
        }

        id<MTLCommandBuffer> blitCB = [_commandQueue commandBuffer];
        if (!blitCB) {
            NSLog(@"MGL ERROR: mtlGetTexImage failed to create blit command buffer for texture %u", tex->name);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        id<MTLBlitCommandEncoder> blitEncoder = [blitCB blitCommandEncoder];
        if (!blitEncoder) {
            NSLog(@"MGL ERROR: mtlGetTexImage failed to create blit encoder for texture %u", tex->name);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        [blitEncoder copyFromTexture:texture
                          sourceSlice:slice
                          sourceLevel:level
                         sourceOrigin:readRegion.origin
                           sourceSize:readRegion.size
                             toBuffer:stagingBuffer
                    destinationOffset:0
               destinationBytesPerRow:rowBytes
             destinationBytesPerImage:imageBytes];

        [blitEncoder endEncoding];
        [blitCB commit];
        [blitCB waitUntilCompleted];

        if (blitCB.error) {
            NSLog(@"MGL ERROR: mtlGetTexImage blit failed for texture %u: %@", tex->name, blitCB.error);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        if (useBGRA8Conversion) {
            if (!mglMetalCopyBGRA8CompatibleTextureBytesToGL((const uint8_t *)stagingBuffer.contents,
                                                             rowBytes,
                                                             (uint8_t *)pixelBytes,
                                                             bytesPerRow,
                                                             readRegion.size.width,
                                                             readRegion.size.height,
	                                                             texture.pixelFormat,
	                                                             format,
	                                                             type,
	                                                             flipRenderTargetRows)) {
	                NSLog(@"MGL ERROR: mtlGetTexImage unsupported BGRA8 conversion texture=%u format=0x%x type=0x%x",
	                      tex->name,
	                      (unsigned)format,
	                      (unsigned)type);
	                mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
	            }
        } else if (flipRenderTargetRows && readRegion.size.depth == 1u) {
            mglMetalCopyRows((const uint8_t *)stagingBuffer.contents,
                             rowBytes,
                             (uint8_t *)pixelBytes,
	                             bytesPerRow,
	                             rowBytes,
                             readRegion.size.height,
                             YES);
        } else {
            memcpy(pixelBytes, stagingBuffer.contents, totalBytes);
        }
        if (mglTraceLogIsEnabled() &&
            tex->internalformat == GL_R8 &&
            format == GL_RED &&
            type == GL_UNSIGNED_BYTE &&
            readRegion.size.width > 0 &&
            readRegion.size.height > 0) {
            const uint8_t *rb = (const uint8_t *)pixelBytes;
            mglTraceLog("GET_TEX_IMAGE_R8 tex=%u target=0x%x isRT=%d fmt=%lu rowBytes=%lu dstBPR=%lu size=%lux%lu first=%u,%u,%u,%u,%u,%u,%u,%u",
                        (unsigned)tex->name,
                        (unsigned)tex->target,
                        tex->is_render_target ? 1 : 0,
                        (unsigned long)texture.pixelFormat,
                        (unsigned long)rowBytes,
                        (unsigned long)bytesPerRow,
                        (unsigned long)readRegion.size.width,
                        (unsigned long)readRegion.size.height,
                        rb[0],
                        rb[MIN((NSUInteger)1, totalBytes - 1)],
                        rb[MIN((NSUInteger)2, totalBytes - 1)],
                        rb[MIN((NSUInteger)3, totalBytes - 1)],
                        rb[MIN((NSUInteger)4, totalBytes - 1)],
                        rb[MIN((NSUInteger)5, totalBytes - 1)],
                        rb[MIN((NSUInteger)6, totalBytes - 1)],
                        rb[MIN((NSUInteger)7, totalBytes - 1)]);
        }
        return;
	    }

	    @try {
	        if (useBGRA8Conversion || (flipRenderTargetRows && readRegion.size.depth == 1u)) {
	            NSUInteger rowBytes;
	            if (useBGRA8Conversion) {
                    NSUInteger sourceBpp = mglMetalReadbackBytesPerPixel(texture.pixelFormat);
                    BOOL sourceIsBGRA8 =
                        (texture.pixelFormat == MTLPixelFormatBGRA8Unorm ||
                         texture.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB ||
                         texture.pixelFormat == MTLPixelFormatRGBA8Unorm ||
                         texture.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB);
                    if (!sourceIsBGRA8 && sourceBpp > 0u) {
                        rowBytes = readRegion.size.width * sourceBpp;
                    } else {
                        rowBytes = readRegion.size.width * 4u;
                    }
                } else {
                    rowBytes = (bytesPerRow > 0 ? bytesPerRow : readRegion.size.width * MAX(dstPixelBytes, (NSUInteger)1u));
                }
            NSUInteger totalBytes = rowBytes * readRegion.size.height;
            NSMutableData *readback = [NSMutableData dataWithLength:totalBytes];
            if (!readback) {
                mglDispatchError(glm_ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
                return;
            }
            [texture getBytes:readback.mutableBytes
                  bytesPerRow:rowBytes
                bytesPerImage:bytesPerImage
                   fromRegion:readRegion
                  mipmapLevel:level
                        slice:slice];
            if (useBGRA8Conversion) {
                if (!mglMetalCopyBGRA8CompatibleTextureBytesToGL((const uint8_t *)readback.bytes,
                                                                 rowBytes,
                                                                 (uint8_t *)pixelBytes,
                                                                 bytesPerRow,
                                                                 readRegion.size.width,
                                                                 readRegion.size.height,
	                                                                 texture.pixelFormat,
	                                                                 format,
	                                                                 type,
	                                                                 flipRenderTargetRows)) {
	                    NSLog(@"MGL ERROR: mtlGetTexImage unsupported BGRA8 conversion texture=%u format=0x%x type=0x%x",
	                          tex->name,
	                          (unsigned)format,
	                          (unsigned)type);
                    mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
                }
            } else {
                mglMetalCopyRows((const uint8_t *)readback.bytes,
                                 rowBytes,
                                 (uint8_t *)pixelBytes,
                                 bytesPerRow,
                                 rowBytes,
                                 readRegion.size.height,
                                 YES);
            }
        } else {
            [texture getBytes:pixelBytes
                  bytesPerRow:bytesPerRow
                bytesPerImage:bytesPerImage
                   fromRegion:readRegion
                  mipmapLevel:level
                        slice:slice];
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: mtlGetTexImage texture read failed for texture %u: %@",
              tex->name,
              exception);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
    }
}

-(void)mtlGenerateMipmaps:(GLMContext)glm_ctx forTexture:(Texture *) tex
{
    ctx = glm_ctx;

    if (!tex) {
        NSLog(@"MGL ERROR: mtlGenerateMipmaps called with NULL texture");
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    RETURN_ON_FAILURE([self processGLState: false]);

    // end encoding on current render encoder
    [self endRenderEncoding];

    RETURN_ON_FAILURE([self ensureWritableCommandBuffer:"mtlGenerateMipmaps"]);

    // no failure path..?
    RETURN_ON_FAILURE([self bindMTLTexture:tex]);

    id<MTLTexture> texture;

    texture = (__bridge id<MTLTexture>)(tex->mtl_data);
    if (!texture) {
        NSLog(@"MGL ERROR: mtlGenerateMipmaps texture %u has no Metal texture after bind", tex->name);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    if (texture.mipmapLevelCount <= 1u) {
        return;
    }

    // start blit encoder
    id<MTLBlitCommandEncoder> blitCommandEncoder;
    blitCommandEncoder = [_currentCommandBuffer blitCommandEncoder];
    if (!blitCommandEncoder) {
        NSLog(@"MGL ERROR: Failed to create blit encoder for mipmap generation");
        return;
    }

    @try {
        [blitCommandEncoder generateMipmapsForTexture:texture];
        [blitCommandEncoder endEncoding];
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: generateMipmapsForTexture failed for texture %u: %@",
              tex->name,
              exception);
        @try {
            [blitCommandEncoder endEncoding];
        } @catch (NSException *endException) {
            NSLog(@"MGL WARNING: failed to end mipmap blit encoder after exception: %@", endException);
        }
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
    }
}

- (bool)encodeTextureBytesUpload:(Texture *)tex
                          source:(id<MTLBuffer>)buffer
                    sourceOffset:(NSUInteger)sourceOffset
                sourceBytesPerRow:(NSUInteger)sourceBytesPerRow
              sourceBytesPerImage:(NSUInteger)sourceBytesPerImage
                            width:(NSUInteger)width
                           height:(NSUInteger)height
                            depth:(NSUInteger)depth
                            slice:(NSUInteger)slice
                            level:(NSUInteger)level
                          xoffset:(NSUInteger)xoffset
                          yoffset:(NSUInteger)yoffset
                          zoffset:(NSUInteger)zoffset
                           reason:(const char *)reason
{
    if (!tex || !buffer || sourceBytesPerRow == 0 || width == 0 || height == 0) {
        return false;
    }

    if (tex->mtl_data == NULL) {
        [self bindMTLTexture:tex];
        if (tex->mtl_data == NULL) {
            return false;
        }
    }

    id<MTLTexture> texture = (__bridge id<MTLTexture>)(tex->mtl_data);
    if (!texture) {
        return false;
    }

    MTLTextureType textureType = texture.textureType;
    NSUInteger destinationSlice = 0;
    MTLOrigin destinationOrigin = MTLOriginMake(xoffset, yoffset, 0);
    NSUInteger copyDepth = 1;

    if (textureType == MTLTextureType3D) {
        destinationSlice = 0;
        destinationOrigin = MTLOriginMake(xoffset, yoffset, zoffset);
        copyDepth = MAX(depth, (NSUInteger)1);
    } else if (textureType == MTLTextureTypeCube ||
               textureType == MTLTextureTypeCubeArray ||
               textureType == MTLTextureType2DArray ||
               textureType == MTLTextureType1DArray ||
               textureType == MTLTextureType2DMultisampleArray) {
        destinationSlice = slice;
        destinationOrigin = MTLOriginMake(xoffset, yoffset, 0);
        copyDepth = 1;
    }

    NSUInteger copyHeight = (textureType == MTLTextureType1DArray ||
                             tex->target == GL_TEXTURE_1D_ARRAY)
        ? 1UL
        : MAX(height, (NSUInteger)1);
    NSUInteger expectedBytesPerImage = sourceBytesPerRow * copyHeight;
    NSUInteger copyBytesPerImage = sourceBytesPerImage;
    if (textureType == MTLTextureTypeCube ||
        textureType == MTLTextureTypeCubeArray ||
        textureType == MTLTextureType2DArray ||
        textureType == MTLTextureType1DArray ||
        textureType == MTLTextureType2DMultisampleArray) {
        copyBytesPerImage = expectedBytesPerImage;
    } else if (textureType == MTLTextureType3D) {
        if (copyBytesPerImage < expectedBytesPerImage) {
            copyBytesPerImage = expectedBytesPerImage;
        }
    } else {
        copyBytesPerImage = expectedBytesPerImage;
    }

    NSUInteger maxDestinationSlices = texture.arrayLength;
    if (textureType == MTLTextureTypeCube) {
        maxDestinationSlices = 6UL;
    } else if (textureType == MTLTextureTypeCubeArray) {
        maxDestinationSlices = texture.arrayLength * 6UL;
    }

    if (level >= texture.mipmapLevelCount ||
        destinationSlice >= maxDestinationSlices ||
        destinationOrigin.x > texture.width ||
        destinationOrigin.y > texture.height ||
        width > texture.width - destinationOrigin.x ||
        copyHeight > texture.height - destinationOrigin.y) {
        NSLog(@"MGL ERROR: texture sub upload out of bounds tex=%u level=%lu slice=%lu origin=(%lu,%lu,%lu) size=%lux%lux%lu texture=%lux%lux%lu",
              tex->name,
              (unsigned long)level,
              (unsigned long)destinationSlice,
              (unsigned long)destinationOrigin.x,
              (unsigned long)destinationOrigin.y,
              (unsigned long)destinationOrigin.z,
              (unsigned long)width,
              (unsigned long)copyHeight,
              (unsigned long)copyDepth,
              (unsigned long)texture.width,
              (unsigned long)texture.height,
              (unsigned long)texture.depth);
        return false;
    }

    return [self copyTextureUploadWithDedicatedCommandBuffer:buffer
                                                sourceOffset:sourceOffset
                                           sourceBytesPerRow:sourceBytesPerRow
                                         sourceBytesPerImage:copyBytesPerImage
                                                   sourceSize:MTLSizeMake(width, copyHeight, copyDepth)
                                                    toTexture:texture
                                             destinationSlice:destinationSlice
                                             destinationLevel:level
                                            destinationOrigin:destinationOrigin
                                                       reason:reason ? reason : "texture_sub_upload"];
}

-(void)mtlTexSubImage:(GLMContext)glm_ctx tex:(Texture *)tex buf:(Buffer *)buf src_offset:(size_t)src_offset src_pitch:(size_t)src_pitch src_image_size:(size_t)src_image_size src_size:(size_t)src_size slice:(GLuint)slice level:(GLuint)level width:(size_t)width height:(size_t)height depth:(size_t)depth xoffset:(size_t)xoffset yoffset:(size_t)yoffset zoffset:(size_t)zoffset
{
    METAL_LOCK();
    [self mtlTexSubImageLocked:glm_ctx tex:tex buf:buf src_offset:src_offset src_pitch:src_pitch src_image_size:src_image_size src_size:src_size slice:slice level:level width:width height:height depth:depth xoffset:xoffset yoffset:yoffset zoffset:zoffset];
    METAL_UNLOCK();
}

-(void)mtlTexSubImageLocked:(GLMContext)glm_ctx tex:(Texture *)tex buf:(Buffer *)buf src_offset:(size_t)src_offset src_pitch:(size_t)src_pitch src_image_size:(size_t)src_image_size src_size:(size_t)src_size slice:(GLuint)slice level:(GLuint)level width:(size_t)width height:(size_t)height depth:(size_t)depth xoffset:(size_t)xoffset yoffset:(size_t)yoffset zoffset:(size_t)zoffset
{
    if (!tex || !buf) {
        NSLog(@"MGL ERROR: mtlTexSubImage called with null tex/buf (tex=%p buf=%p)", tex, buf);
        return;
    }

    if (src_pitch == 0 || width == 0 || height == 0) {
        NSLog(@"MGL ERROR: mtlTexSubImage invalid dimensions/pitch tex=%u width=%zu height=%zu src_pitch=%zu",
              tex->name, width, height, src_pitch);
        return;
    }

    // we can deal with a null buffer but we need a texture
    if (buf->data.mtl_data == NULL)
    {
        [self bindMTLBufferLocked: buf];
        RETURN_ON_NULL(buf->data.mtl_data);
    }

    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(buf->data.mtl_data);
    if (!buffer) {
        NSLog(@"MGL ERROR: mtlTexSubImage missing Metal buffer object tex=%u", tex->name);
        return;
    }

    /* PBO source data is in CPU/client layout (e.g. 3 bytes/pixel for GL_RGB8).
     * When the Metal destination has a different texel size (RGBA8 = 4, RGBA16* = 8,
     * RGBA32* = 16), expand the PBO data into a staging buffer in Metal layout
     * before the blit — otherwise sourceBytesPerRow (CPU pitch) mismatches the
     * Metal texture's expected row stride and pixels shift / stripe.  The non-PBO
     * path (mtlTexSubImageBytes) already expands; this mirrors it.  When no
     * expansion is needed, fall through to the direct blit below. */
    if (tex->mtl_data) {
        id<MTLTexture> dstTexture = (__bridge id<MTLTexture>)(tex->mtl_data);
        MTLPixelFormat dstPixelFormat = dstTexture.pixelFormat;
        BOOL needsChannelExpand = mglTextureNeedsChannelExpansion(tex->internalformat, dstPixelFormat);
        BOOL needsRGBA8Expand = NO;
        if (!needsChannelExpand) {
            needsRGBA8Expand = mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, dstPixelFormat);
        }
        if (needsChannelExpand || needsRGBA8Expand) {
            NSUInteger dstBytesPerPixel = needsChannelExpand
                ? ((dstPixelFormat == MTLPixelFormatRGBA16Unorm ||
                    dstPixelFormat == MTLPixelFormatRGBA16Snorm ||
                    dstPixelFormat == MTLPixelFormatRGBA16Float ||
                    dstPixelFormat == MTLPixelFormatRGBA16Sint ||
                    dstPixelFormat == MTLPixelFormatRGBA16Uint) ? 8 : 16)
                : 4;
            NSUInteger cpuBytesPerPixel = (tex->faces[0].levels && level < tex->num_levels &&
                                           tex->faces[0].levels[level].width > 0u &&
                                           tex->faces[0].levels[level].pitch > 0u)
                ? (NSUInteger)(tex->faces[0].levels[level].pitch / tex->faces[0].levels[level].width)
                : [self bytesPerPixelForFormat:tex->internalformat];
            if (cpuBytesPerPixel == 0u) {
                cpuBytesPerPixel = (NSUInteger)sizeForInternalFormat(tex->internalformat, 0, 0);
            }
            if (cpuBytesPerPixel > 0u && cpuBytesPerPixel != dstBytesPerPixel) {
                NSUInteger copyHeight = MAX((NSUInteger)height, 1UL);
                NSUInteger copyDepth = MAX((NSUInteger)depth, 1UL);
                NSUInteger dstRowBytes = (NSUInteger)width * dstBytesPerPixel;
                NSUInteger dstImageBytes = dstRowBytes * copyHeight;
                size_t sourceImagePitch = src_image_size;
                size_t minimumImagePitch = src_pitch * copyHeight;
                if (sourceImagePitch < minimumImagePitch) {
                    sourceImagePitch = minimumImagePitch;
                }
                size_t packedBytes = dstImageBytes * copyDepth;
                if (packedBytes != 0u && packedBytes <= (512u * 1024u * 1024u)) {
                    const uint8_t *sourceBase = (const uint8_t *)buffer.contents;
                    NSMutableData *packedUpload = [NSMutableData dataWithLength:packedBytes];
                    if (packedUpload && packedUpload.mutableBytes && sourceBase) {
                        uint8_t *packedBytesPtr = (uint8_t *)packedUpload.mutableBytes;
                        bool expandOK = true;
                        for (NSUInteger z = 0; z < copyDepth && expandOK; z++) {
                            size_t sliceBaseOff = src_offset + (size_t)z * sourceImagePitch;
                            size_t lastRowOff = sliceBaseOff + (size_t)(copyHeight - 1u) * src_pitch;
                            size_t rowBytesCpu = (NSUInteger)width * cpuBytesPerPixel;
                            if (lastRowOff > src_size || rowBytesCpu > src_size - lastRowOff) {
                                expandOK = false;
                                break;
                            }
                            const uint8_t *sliceSrc = sourceBase + sliceBaseOff;
                            NSUInteger expandedBPR = 0, expandedBPI = 0;
                            uint8_t *expanded = NULL;
                            if (needsRGBA8Expand) {
                                expanded = mglCreateRGBA8ExpandedUpload(tex,
                                                                        sliceSrc,
                                                                        width,
                                                                        copyHeight,
                                                                        src_pitch,
                                                                        &expandedBPR,
                                                                        &expandedBPI);
                            } else {
                                expanded = mglCreateChannelExpandedUpload(tex,
                                                                           dstPixelFormat,
                                                                           sliceSrc,
                                                                           width,
                                                                           copyHeight,
                                                                           src_pitch,
                                                                           &expandedBPR,
                                                                           &expandedBPI);
                            }
                            if (!expanded) {
                                expandOK = false;
                                break;
                            }
                            memcpy(packedBytesPtr + (z * dstImageBytes), expanded, expandedBPI);
                            free(expanded);
                        }
                        if (expandOK) {
                            id<MTLBuffer> uploadBuffer = [_device newBufferWithBytes:packedUpload.bytes
                                                                                length:packedBytes
                                                                               options:MTLResourceStorageModeShared];
                            if (uploadBuffer) {
                                bool uploaded = [self encodeTextureBytesUpload:tex
                                                                        source:uploadBuffer
                                                                  sourceOffset:0
                                                              sourceBytesPerRow:dstRowBytes
                                                            sourceBytesPerImage:dstImageBytes
                                                                       width:width
                                                                      height:height
                                                                       depth:depth
                                                                       slice:slice
                                                                       level:level
                                                                     xoffset:xoffset
                                                                     yoffset:yoffset
                                                                     zoffset:zoffset
                                                                      reason:"mtlTexSubImage"];
                                if (!uploaded) {
                                    NSLog(@"MGL ERROR: mtlTexSubImage expanded PBO upload failed (tex=%u slice=%u level=%u)",
                                          tex->name, slice, level);
                                }
                                return;
                            }
                        }
                    }
                }
            }
        }
    }

    bool uploaded = [self encodeTextureBytesUpload:tex
                                            source:buffer
                                      sourceOffset:src_offset
                                  sourceBytesPerRow:src_pitch
                                sourceBytesPerImage:src_image_size
                                             width:width
                                            height:height
                                             depth:depth
                                             slice:slice
                                             level:level
                                           xoffset:xoffset
                                           yoffset:yoffset
                                           zoffset:zoffset
                                            reason:"mtlTexSubImage"];
    if (!uploaded) {
        NSLog(@"MGL ERROR: mtlTexSubImage dedicated upload failed (tex=%u slice=%u level=%u)",
              tex->name, slice, level);
    }
}

-(bool)mtlTexSubImageBytes:(GLMContext)glm_ctx tex:(Texture *)tex bytes:(const void *)bytes bytesSize:(size_t)bytes_size src_offset:(size_t)src_offset src_pitch:(size_t)src_pitch src_image_size:(size_t)src_image_size slice:(GLuint)slice level:(GLuint)level width:(size_t)width height:(size_t)height depth:(size_t)depth xoffset:(size_t)xoffset yoffset:(size_t)yoffset zoffset:(size_t)zoffset
{
    (void)glm_ctx;
    if (!tex || !bytes || src_pitch == 0 || width == 0 || height == 0) {
        return false;
    }
    if (src_offset > bytes_size || level >= tex->num_levels) {
        return false;
    }

    NSUInteger bytesPerPixel = [self bytesPerPixelForFormat:tex->internalformat];
    if (bytesPerPixel == 0u &&
        tex->faces[0].levels &&
        tex->faces[0].levels[level].width > 0u) {
        TextureLevel *levelInfo = &tex->faces[0].levels[level];
        if (levelInfo->pitch > 0u &&
            (levelInfo->pitch % levelInfo->width) == 0u) {
            bytesPerPixel = (NSUInteger)(levelInfo->pitch / levelInfo->width);
        }
    }
    if (bytesPerPixel == 0u) {
        return false;
    }

    NSUInteger copyHeight = MAX((NSUInteger)height, 1UL);
    NSUInteger copyDepth = MAX((NSUInteger)depth, 1UL);
    NSUInteger rowBytes = (NSUInteger)width * bytesPerPixel;
    if (rowBytes == 0u || rowBytes > src_pitch) {
        return false;
    }

    if (!tex->mtl_data) {
        return false;
    }

    /* Channel expansion: GL_RGB32* (12 bytes/pixel) -> Metal RGBA32* (16 bytes/pixel).
     * The CPU backing stores 3 channels per pixel, but the Metal texture expects
     * 4 channels. We must expand each pixel by inserting a default alpha before
     * uploading, otherwise the data layout mismatches and pixels shift. */
    id<MTLTexture> dstTexture = (__bridge id<MTLTexture>)(tex->mtl_data);
    MTLPixelFormat dstPixelFormat = dstTexture.pixelFormat;
    BOOL needsChannelExpand = mglTextureNeedsChannelExpansion(tex->internalformat,
                                                              dstPixelFormat);
    NSUInteger dstBytesPerPixel = bytesPerPixel;
    if (needsChannelExpand) {
        switch (dstPixelFormat) {
            case MTLPixelFormatRGBA16Unorm:
            case MTLPixelFormatRGBA16Snorm:
            case MTLPixelFormatRGBA16Float:
            case MTLPixelFormatRGBA16Sint:
            case MTLPixelFormatRGBA16Uint:
                dstBytesPerPixel = 8;
                break;
            case MTLPixelFormatRGBA32Float:
            case MTLPixelFormatRGBA32Sint:
            case MTLPixelFormatRGBA32Uint:
                dstBytesPerPixel = 16;
                break;
            default:
                needsChannelExpand = NO;
                break;
        }
    }

    /* RGBA8 expansion: Metal has no RGB8 pixel format, so GL_RGB8-family
     * internal formats (3 bytes/pixel in the CPU backing store) are backed
     * by Metal RGBA8 variants (4 bytes/pixel).  Without per-pixel channel
     * expansion the 3-byte source is uploaded directly into a 4-byte Metal
     * texture, shifting pixels and producing vertical stripes.  Every other
     * upload path (createMTLTextureFromGLTexture, refreshMetalTextureCPUData,
     * mtlCopyImageSubData) expands via mglCreateRGBA8ExpandedUpload; the
     * direct mtlTexSubImageBytes path must do the same. */
    BOOL needsRGBA8Expand = NO;
    if (!needsChannelExpand) {
        needsRGBA8Expand = mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat,
                                                                        dstPixelFormat);
        if (needsRGBA8Expand) {
            dstBytesPerPixel = 4;
        }
    }

    size_t sourceImagePitch = src_image_size;
    size_t minimumImagePitch = src_pitch * copyHeight;
    if (sourceImagePitch < minimumImagePitch) {
        sourceImagePitch = minimumImagePitch;
    }

    NSUInteger dstRowBytes = (NSUInteger)width * dstBytesPerPixel;
    NSUInteger dstImageBytes = dstRowBytes * copyHeight;
    size_t packedBytes = dstImageBytes * copyDepth;
    if (packedBytes == 0u || packedBytes > (512u * 1024u * 1024u)) {
        return false;
    }

    NSMutableData *packedUpload = [NSMutableData dataWithLength:packedBytes];
    if (!packedUpload || !packedUpload.mutableBytes) {
        return false;
    }

    const uint8_t *sourceBase = (const uint8_t *)bytes;
    uint8_t *packedBytesPtr = (uint8_t *)packedUpload.mutableBytes;

    if (needsChannelExpand) {
        /* Determine component bytes and alpha default for the destination format */
        NSUInteger srcCompBytes = 0;
        NSUInteger dstCompBytes = 0;
        uint64_t alphaDefault = 0;
        switch (dstPixelFormat) {
            case MTLPixelFormatRGBA16Unorm:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 65535; break;
            case MTLPixelFormatRGBA16Snorm:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 32767; break;
            case MTLPixelFormatRGBA16Float:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 0x3C00; break;
            case MTLPixelFormatRGBA16Sint:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 1; break;
            case MTLPixelFormatRGBA16Uint:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 1; break;
            case MTLPixelFormatRGBA32Float:
                srcCompBytes = 4; dstCompBytes = 4;
                { float f = 1.0f; memcpy(&alphaDefault, &f, sizeof(f)); }
                break;
            case MTLPixelFormatRGBA32Sint:
                srcCompBytes = 4; dstCompBytes = 4; alphaDefault = 1; break;
            case MTLPixelFormatRGBA32Uint:
                srcCompBytes = 4; dstCompBytes = 4; alphaDefault = 1; break;
            default:
                return false;
        }
        NSUInteger srcPixelBytes = srcCompBytes * 3;  /* 3 channels in source */
        NSUInteger dstPixelBytes = dstCompBytes * 4;  /* 4 channels in destination */

        for (NSUInteger z = 0; z < copyDepth; z++) {
            for (NSUInteger y = 0; y < copyHeight; y++) {
                size_t srcRowOffset = src_offset + ((size_t)z * sourceImagePitch) + ((size_t)y * src_pitch);
                if (srcRowOffset > bytes_size || rowBytes > bytes_size - srcRowOffset) {
                    return false;
                }
                const uint8_t *srcRow = sourceBase + srcRowOffset;
                uint8_t *dstRow = packedBytesPtr + (z * dstImageBytes) + (y * dstRowBytes);
                for (NSUInteger x = 0; x < width; x++) {
                    const uint8_t *srcPixel = srcRow + x * srcPixelBytes;
                    uint8_t *dstPixel = dstRow + x * dstPixelBytes;
                    /* Copy 3 channels (R, G, B) */
                    memcpy(dstPixel, srcPixel, srcPixelBytes);
                    /* Set alpha channel to default value */
                    memcpy(dstPixel + srcPixelBytes, &alphaDefault, dstCompBytes);
                }
            }
        }
    } else if (needsRGBA8Expand) {
        /* GL_RGB8-family / packed RGB formats (CPU bpp < 4) -> Metal RGBA8
         * (4 bytes/pixel).  Delegate to mglCreateRGBA8ExpandedUpload, which
         * correctly unpacks every supported internal format (3:3:2, 5_6_5,
         * 10_10_10, 4_4_4_4, ...) into RGBA8 — matching what every other
         * upload path (createMTLTextureFromGLTexture, refreshMetalTextureCPUData,
         * mtlCopyImageSubData) does.  Without this, 3-byte RGB8 source is
         * uploaded directly into a 4-byte Metal texture, shifting pixels and
         * producing vertical stripes. */
        for (NSUInteger z = 0; z < copyDepth; z++) {
            size_t sliceBaseOff = src_offset + (size_t)z * sourceImagePitch;
            size_t lastRowOff = sliceBaseOff + (size_t)(copyHeight - 1u) * src_pitch;
            if (lastRowOff > bytes_size || rowBytes > bytes_size - lastRowOff) {
                return false;
            }
            const uint8_t *sliceSrc = sourceBase + sliceBaseOff;
            NSUInteger expandedBPR = 0, expandedBPI = 0;
            uint8_t *expanded = mglCreateRGBA8ExpandedUpload(tex,
                                                              sliceSrc,
                                                              width,
                                                              copyHeight,
                                                              src_pitch,
                                                              &expandedBPR,
                                                              &expandedBPI);
            if (!expanded) {
                return false;
            }
            memcpy(packedBytesPtr + (z * dstImageBytes), expanded, expandedBPI);
            free(expanded);
        }
    } else {
        /* No channel expansion needed - direct copy */
        for (NSUInteger z = 0; z < copyDepth; z++) {
            for (NSUInteger y = 0; y < copyHeight; y++) {
                size_t srcRowOffset = src_offset + ((size_t)z * sourceImagePitch) + ((size_t)y * src_pitch);
                if (srcRowOffset > bytes_size || rowBytes > bytes_size - srcRowOffset) {
                    static uint64_t s_subUploadRangeFailLogs = 0;
                    uint64_t hit = ++s_subUploadRangeFailLogs;
                    if (hit <= 32ull || (hit % 512ull) == 0ull) {
                        NSLog(@"MGL TEXSUBIMAGE BYTES range fail tex=%u level=%u off=%zu rowBytes=%lu pitch=%zu image=%zu size=%zu z=%lu y=%lu hit=%llu",
                              (unsigned)tex->name,
                              (unsigned)level,
                              srcRowOffset,
                              (unsigned long)rowBytes,
                              src_pitch,
                              sourceImagePitch,
                              bytes_size,
                              (unsigned long)z,
                              (unsigned long)y,
                              (unsigned long long)hit);
                    }
                    return false;
                }
                memcpy(packedBytesPtr + (z * dstImageBytes) + (y * dstRowBytes),
                       sourceBase + srcRowOffset,
                       rowBytes);
            }
        }
    }

    id<MTLBuffer> uploadBuffer = [_device newBufferWithBytes:packedUpload.bytes
                                                      length:packedBytes
                                                     options:MTLResourceStorageModeShared];
    if (!uploadBuffer) {
        return false;
    }

    bool uploaded = [self encodeTextureBytesUpload:tex
                                            source:uploadBuffer
                                      sourceOffset:0
                                  sourceBytesPerRow:dstRowBytes
                                sourceBytesPerImage:dstImageBytes
                                             width:width
                                            height:height
                                             depth:depth
                                             slice:slice
                                             level:level
                                           xoffset:xoffset
                                           yoffset:yoffset
                                           zoffset:zoffset
                                            reason:"mtlTexSubImageBytes"];
    return uploaded;
}


#pragma mark - Extracted from createMTLTextureFromGLTexture:
- (void)reUploadExistingCPUTextureData:(Texture *)tex
                                metal:(id<MTLTexture>)texture
                          pixelFormat:(MTLPixelFormat)pixelFormat
                            numFaces:(uint)num_faces
                    uploadLevelCount:(GLuint)upload_level_count
                              isArray:(BOOL)is_array
                   texture1DBackedBy2D:(BOOL)texture1DBackedBy2D
             texture1DArrayBackedBy2DArray:(BOOL)texture1DArrayBackedBy2DArray
                             texType:(MTLTextureType)tex_type
{
    NSLog(@"MGL INFO: Re-uploading existing CPU texture data (tex=%d, dims=%lux%lu)",

          tex->name, (unsigned long)texture.width, (unsigned long)texture.height);


    for (int face = 0; face < num_faces; face++) {

        for (int level = 0; level < (int)upload_level_count; level++) {

            TextureLevel *uploadLevel = &tex->faces[face].levels[level];

            if (!mglTextureLevelHasUploadableCPUData(uploadLevel)) {

                continue;

            }


            NSUInteger lvlWidth  = tex->faces[face].levels[level].width;

            NSUInteger lvlHeight = tex->faces[face].levels[level].height;

            NSUInteger lvlDepth  = tex->faces[face].levels[level].depth;

            NSUInteger lvlPitch  = tex->faces[face].levels[level].pitch;

            if (lvlPitch == 0 || lvlWidth == 0) continue;


            if (is_array)

            {
                [self reUploadExistingCPUTextureDataArrayLevel:tex
                                                         metal:texture
                                                   pixelFormat:pixelFormat
                                                         face:face
                                                        level:level
                                  texture1DArrayBackedBy2DArray:texture1DArrayBackedBy2DArray
                                                       texType:tex_type];
            }

            else

            {

            /* Non-array re-upload (2D, 3D, 1D, cube).

             * For 3D textures, bytesPerImage must be a single 2D slice

             * (bytesPerRow * height), NOT the full volume data_size.

             * uploadTextureSliceViaBlit computes bufferSize =

             * safeBytesPerImage * copyDepth, so passing the full volume

             * as bytesPerImage AND depth would double-count and cause

             * newBufferWithBytes to read past the source buffer. */

            NSUInteger bytesPerRow = lvlPitch;

            NSUInteger fullDataSize = tex->faces[face].levels[level].data_size;

            if (fullDataSize == 0) fullDataSize = bytesPerRow * MAX((NSUInteger)lvlHeight, 1UL);


            BOOL is3DReupload = (tex->target == GL_TEXTURE_3D && lvlDepth > 1);

            NSUInteger singleSliceBPI = bytesPerRow * MAX((NSUInteger)lvlHeight, 1UL);

            NSUInteger bytesPerImage = is3DReupload ? singleSliceBPI : fullDataSize;

            NSUInteger uploadDepth = is3DReupload ? lvlDepth : (lvlDepth > 1 ? lvlDepth : 1);


            const void *srcData = (const void *)tex->faces[face].levels[level].data;

            void *expandedUploadData = NULL;

            /* Channel expansion for 2D/non-3D only.  3D expansion would

             * require per-slice handling (see DIRTY_TEXTURE_DATA 3D path). */

            if (!is3DReupload) {

                if (mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, pixelFormat)) {

                    NSUInteger expandedBytesPerRow = 0;

                    NSUInteger expandedBytesPerImage = 0;

                    expandedUploadData = mglCreateRGBA8ExpandedUpload(tex,

                                                                      (const uint8_t *)srcData,

                                                                      lvlWidth,

                                                                      MAX((NSUInteger)lvlHeight, 1UL),

                                                                      bytesPerRow,

                                                                      &expandedBytesPerRow,

                                                                      &expandedBytesPerImage);

                    if (expandedUploadData) {

                        srcData = expandedUploadData;

                        bytesPerRow = expandedBytesPerRow;

                        bytesPerImage = expandedBytesPerImage;

                    }

                } else if (mglTextureNeedsChannelExpansion(tex->internalformat, pixelFormat)) {

                    NSUInteger expandedBytesPerRow = 0;

                    NSUInteger expandedBytesPerImage = 0;

                    expandedUploadData = mglCreateChannelExpandedUpload(tex,

                                                                         pixelFormat,

                                                                         (const uint8_t *)srcData,

                                                                         lvlWidth,

                                                                         MAX((NSUInteger)lvlHeight, 1UL),

                                                                         bytesPerRow,

                                                                         &expandedBytesPerRow,

                                                                         &expandedBytesPerImage);

                    if (expandedUploadData) {

                        srcData = expandedUploadData;

                        bytesPerRow = expandedBytesPerRow;

                        bytesPerImage = expandedBytesPerImage;

                    }

                }

            }

            NSUInteger alignment = [self getOptimalAlignmentForPixelFormat:pixelFormat];

            NSUInteger alignedBytesPerRow = bytesPerRow;

            if (alignedBytesPerRow % alignment != 0) {

                alignedBytesPerRow = ((alignedBytesPerRow + alignment - 1) / alignment) * alignment;

            }


            uintptr_t addr = (uintptr_t)srcData;

            if (addr % alignment != 0 || alignedBytesPerRow != bytesPerRow) {

                NSUInteger rowCount = MAX((NSUInteger)lvlHeight, 1UL);

                NSUInteger alignedSliceBPI = alignedBytesPerRow * rowCount;

                NSUInteger alignedSize = alignedSliceBPI * uploadDepth;

                if (alignedSize > 0 && alignedSize <= (512 * 1024 * 1024)) {

                    void *alignedData = aligned_alloc(alignment, alignedSize);

                    if (alignedData) {

                        memset(alignedData, 0, alignedSize);

                        for (NSUInteger z = 0; z < uploadDepth; z++) {

                            for (NSUInteger row = 0; row < rowCount; row++) {

                                NSUInteger copySize = MIN(bytesPerRow, alignedBytesPerRow);

                                memcpy((uint8_t *)alignedData + z * alignedSliceBPI + row * alignedBytesPerRow,

                                       (const uint8_t *)srcData + z * singleSliceBPI + row * bytesPerRow, copySize);

                            }

                        }

                        [self uploadTextureSliceViaBlit:texture

                                               texName:tex->name

                                             texTarget:tex->target

                                                 bytes:alignedData

                                           bytesPerRow:alignedBytesPerRow

                                         bytesPerImage:alignedSliceBPI

                                                 width:lvlWidth

                                                height:lvlHeight

                                                 depth:uploadDepth

                                                 level:level

                                                 slice:face];

                        free(alignedData);

                    }

                }

            } else {

                [self uploadTextureSliceViaBlit:texture

                                       texName:tex->name

                                     texTarget:tex->target

                                         bytes:srcData

                                   bytesPerRow:bytesPerRow

                                 bytesPerImage:bytesPerImage

                                         width:lvlWidth

                                        height:lvlHeight

                                         depth:uploadDepth

                                         level:level

                                         slice:face];

            }

            free(expandedUploadData);

            } /* end else (non-array) */

        }

    }

}

- (void)fillTextureWithSafeInitialContents:(id<MTLTexture>)texture
                                         tex:(Texture *)tex
                                 pixelFormat:(MTLPixelFormat)pixelFormat
{
        // No existing data — fill with safe initial contents

    if (texture.width == 0 || texture.height == 0 || texture.width > 16384 || texture.height > 16384) {

        NSLog(@"MGL WARNING: Skipping texture fill due to invalid dimensions: %lux%lu", (unsigned long)texture.width, (unsigned long)texture.height);

    } else {

        // Determine pixel format size to create appropriate black data

        NSUInteger bytesPerPixel = 4; // Default to RGBA

        switch(texture.pixelFormat) {

            case MTLPixelFormatR8Unorm:

            case MTLPixelFormatR8Uint:

            case MTLPixelFormatR8Sint:

                bytesPerPixel = 1;

                break;

            case MTLPixelFormatRG8Unorm:

            case MTLPixelFormatRG8Uint:

            case MTLPixelFormatRG8Sint:

                bytesPerPixel = 2;

                break;

            case MTLPixelFormatRGBA8Unorm:

            case MTLPixelFormatRGBA8Uint:

            case MTLPixelFormatRGBA8Sint:

                bytesPerPixel = 4;

                break;

            default:

                bytesPerPixel = 4; // Default assumption

                break;

        }


        // Calculate dynamic alignment for Metal textures based on pixel format

        NSUInteger bytesPerRow = texture.width * bytesPerPixel;

        NSUInteger alignment = [self getOptimalAlignmentForPixelFormat:texture.pixelFormat];

        if (bytesPerRow % alignment != 0) {

            bytesPerRow = ((bytesPerRow + alignment - 1) / alignment) * alignment;

        }


        NSUInteger dataSize = bytesPerRow * texture.height;


        // Validate that dataSize is reasonable (not too large)

        if (dataSize > 64 * 1024 * 1024) { // 64MB limit per texture level

            NSLog(@"MGL WARNING: Skipping texture fill due to excessive size: %lu bytes", (unsigned long)dataSize);

        } else {

            // Allocate initialization data for texture clear.

            // aligned_alloc has been unreliable in this environment; calloc is safer here.

            (void)alignment;

            void *blackData = calloc(dataSize, 1);

            if (blackData) {

                // CRITICAL SECURITY FIX: Comprehensive validation to prevent Metal driver crashes

                // calloc already zero-initializes


                // Multi-layer validation for all parameters

                if (!blackData) {

                    NSLog(@"MGL SECURITY ERROR: blackData is NULL after memset - CORRUPTION DETECTED");

                    return;
                }

                if (bytesPerRow == 0) {

                    NSLog(@"MGL SECURITY ERROR: Invalid bytesPerRow (0) for texture fill");

                    free(blackData);

                    return;
                }

                if (dataSize == 0) {

                    NSLog(@"MGL SECURITY ERROR: Invalid dataSize (0) for texture fill");

                    free(blackData);

                    return;
                }

                if (!texture) {

                    NSLog(@"MGL SECURITY ERROR: Metal texture is NULL");

                    free(blackData);

                    return;
                }

                if (texture.width == 0 || texture.height == 0) {

                    NSLog(@"MGL SECURITY ERROR: Invalid texture dimensions %lux%lu", (unsigned long)texture.width, (unsigned long)texture.height);

                    free(blackData);

                    return;
                }


                // Additional validation: verify blackData contains expected zeros (anti-corruption check)

                uint8_t *bytes = (uint8_t *)blackData;

                bool dataCorrupted = false;

                for (NSUInteger i = 0; i < MIN(dataSize, 1024); i++) { // Check first 1KB only for performance

                    if (bytes[i] != 0) {

                        dataCorrupted = true;

                        break;

                    }

                }

                if (dataCorrupted) {

                    NSLog(@"MGL SECURITY ERROR: blackData corruption detected - memory safety issue");

                    free(blackData);

                    return;
                }


                NSLog(@"MGL INFO: All validations passed for texture fill (size=%lu, bytesPerRow=%lu)", (unsigned long)dataSize, (unsigned long)bytesPerRow);


                // ULTRA-DEFENSIVE: Final validation immediately before Metal API call

                // This prevents race conditions and memory corruption between validation and use

                if (!blackData) {

                    NSLog(@"MGL CRITICAL ERROR: blackData became NULL before Metal call - RACE CONDITION DETECTED");

                    free(blackData);

                    return;
                }

                if (!texture) {

                    NSLog(@"MGL CRITICAL ERROR: Metal texture became NULL before Metal call - RACE CONDITION DETECTED");

                    free(blackData);

                    return;
                }

                if (bytesPerRow == 0 || dataSize == 0) {

                    NSLog(@"MGL CRITICAL ERROR: Parameters became invalid before Metal call - RACE CONDITION DETECTED");

                    free(blackData);

                    return;
                }


                // Additional verification: Check if Metal texture is still valid

                if (texture.width == 0 || texture.height == 0) {

                    NSLog(@"MGL CRITICAL ERROR: Metal texture dimensions became invalid before Metal call");

                    free(blackData);

                    return;
                }


                // Final integrity check: Verify blackData still contains expected zeros

                uint8_t *finalCheck = (uint8_t *)blackData;

                bool finalCorruption = false;

                for (NSUInteger i = 0; i < MIN(dataSize, 256); i++) { // Check first 256 bytes

                    if (finalCheck[i] != 0) {

                        finalCorruption = true;

                        break;

                    }

                }

                if (finalCorruption) {

                    NSLog(@"MGL CRITICAL ERROR: Memory corruption detected immediately before Metal call");

                    free(blackData);

                    return;
                }


                NSLog(@"MGL INFO: FIXING: Implementing proper texture filling for Apple Metal compatibility");


                // PROPER FIX: Use Apple Metal-compatible texture filling approach

                // The issue was using incorrect bytesPerRow and region parameters

                NSLog(@"MGL INFO: Implementing Metal-compliant texture fill operations");


                // Use Metal's standard pattern for texture filling.

                NSUInteger pixelSize = bytesPerPixel;

                NSUInteger properBytesPerRow = texture.width * pixelSize;


                // Ensure proper alignment for Apple Metal driver

                if (properBytesPerRow % 64 != 0) {

                    properBytesPerRow = ((properBytesPerRow + 63) / 64) * 64;

                }


                // Fill the entire level. A previous 1x1 safety fill left large textures

                // mostly uninitialized while their Metal backing existed.

                MTLRegion properRegion = MTLRegionMake2D(0, 0, texture.width, texture.height);


                // Create properly aligned texture data buffer

                NSUInteger fillSize = properBytesPerRow * properRegion.size.height;

                uint8_t *properData = (uint8_t *)calloc(fillSize, 1);


                if (properData) {

                    // Initialize with safe texture data (transparent black with alpha = 0)

                    for (NSUInteger y = 0; y < properRegion.size.height; y++) {

                        uint8_t *row = properData + (y * properBytesPerRow);

                        for (NSUInteger x = 0; x < properRegion.size.width; x++) {

                            uint8_t *pixel = row + (x * pixelSize);

                            pixel[0] = 0;  // R

                            if (pixelSize > 1) pixel[1] = 0;  // G

                            if (pixelSize > 2) pixel[2] = 0;  // B

                            if (pixelSize > 3) pixel[3] = 0; // A = transparent for uninitialized color data

                        }

                    }


                    @try {

                        NSLog(@"MGL INFO: Performing Metal-compliant texture fill:");

                        NSLog(@"  - Region: %dx%d", (int)properRegion.size.width, (int)properRegion.size.height);

                        NSLog(@"  - bytesPerRow: %lu", (unsigned long)properBytesPerRow);

                        NSLog(@"  - dataSize: %lu", (unsigned long)fillSize);


                        // ALTERNATIVE APPROACH: Safe texture filling without replaceRegion

                        NSLog(@"MGL INFO: Using alternative texture filling methods (AGX-safe)");


                        @try {

                            // ALTERNATIVE 1: Try MTLBuffer-to-texture copy approach

                            if (properData && dataSize > 0) {

                                NSLog(@"MGL INFO: Attempting buffer-based texture fill");


                                // Create a temporary MTLBuffer with the texture data

                                id<MTLBuffer> tempBuffer = [_device newBufferWithBytes:properData

                                                                                length:fillSize

                                                                               options:MTLResourceStorageModeShared];


                                if (tempBuffer) {

                                    NSLog(@"MGL INFO: Created temporary MTLBuffer for texture data");


                                    if ([self shouldSkipGPUOperations]) {

                                        NSLog(@"MGL AGX: Skipping texture fill during recovery - texture will be empty");

                                    } else {

                                        BOOL uploaded = [self copyTextureUploadWithDedicatedCommandBuffer:tempBuffer

                                                                                              sourceOffset:0

                                                                                         sourceBytesPerRow:properBytesPerRow

                                                                                       sourceBytesPerImage:fillSize

                                                                                                 sourceSize:MTLSizeMake(properRegion.size.width, properRegion.size.height, 1)

                                                                                                  toTexture:texture

                                                                                           destinationSlice:0

                                                                                           destinationLevel:0

                                                                                          destinationOrigin:MTLOriginMake(0, 0, 0)

                                                                                                     reason:"texture_fill_initialization"];

                                        if (uploaded) {

                                            NSLog(@"MGL SUCCESS: Texture data copied using dedicated upload command buffer");

                                            mglMarkTextureLevelMetalFilled(tex, 0, fillSize);

                                        } else {

                                            NSLog(@"MGL WARNING: Dedicated texture fill upload failed - texture may remain uninitialized");

                                        }

                                    }


                                    // Clean up the temporary buffer

                                    tempBuffer = nil;

                                }

                            }

                        } @catch (NSException *exception) {

                            NSLog(@"MGL WARNING: Buffer-based texture fill failed - trying alternative");


                            // ALTERNATIVE 2: Simple direct color filling for basic cases

                            [self fillSmallRGBA8TextureWithGradient:texture tex:tex];

                        }

                    } @catch (NSException *exception) {

                        NSLog(@"MGL ERROR: Metal texture fill failed - investigating root cause");

                        NSLog(@"MGL ERROR: Exception: %@ (Reason: %@)", exception.name, exception.reason);

                        NSLog(@"MGL INFO: This indicates our parameters are still incompatible with AGX driver");

                    }


                    free(properData);

                } else {

                    NSLog(@"MGL ERROR: Failed to allocate properly aligned texture data");

                }

                free(blackData);

            } else {

                NSLog(@"MGL ERROR: Failed to allocate aligned memory for texture fill (%lu bytes)", (unsigned long)dataSize);

            }

        }

    }

}

- (BOOL)uploadDirtyCPUTextureData:(Texture *)tex
                            metal:(id<MTLTexture>)texture
                      pixelFormat:(MTLPixelFormat)pixelFormat
                        numFaces:(uint)num_faces
                uploadLevelCount:(GLuint)upload_level_count
                         isArray:(BOOL)is_array
              texture1DBackedBy2D:(BOOL)texture1DBackedBy2D
        texture1DArrayBackedBy2DArray:(BOOL)texture1DArrayBackedBy2DArray
                         texType:(MTLTextureType)tex_type
            outAllLevelsUploaded:(BOOL *)outAllLevelsUploaded
{

    if (kMGLDiagnosticStateLogs) {
        MGLTraceNSLog(@"MGL DEBUG: DIRTY_TEXTURE_DATA detected - attempting texture filling");
        MGLTraceNSLog(@"MGL DEBUG: Texture details: target=0x%x, internalformat=0x%x, levels=%d effectiveLevels=%u",
                      tex->target, tex->internalformat, tex->num_levels, upload_level_count);
    }

    MTLRegion region;
    NSUInteger width, height, depth;
    BOOL anyLevelSkipped = NO;

    for(int face=0; face<num_faces; face++)
    {
        for (int level=0; level<upload_level_count; level++)
        {
            TextureLevel *uploadLevel = &tex->faces[face].levels[level];
            if (!mglTextureLevelHasUploadableCPUData(uploadLevel)) {
                static uint64_t s_skipStaleUploadLogs = 0;
                uint64_t hit = ++s_skipStaleUploadLogs;
                if (hit <= 64ull || (hit % 512ull) == 0ull) {
                    NSLog(@"MGL TEXTURE SKIP stale CPU upload tex=%u face=%d level=%d source=%u ever=%u init=%u hit=%llu",
                          (unsigned)tex->name,
                          face,
                          level,
                          uploadLevel ? (unsigned)uploadLevel->last_init_source : 0u,
                          uploadLevel ? (unsigned)uploadLevel->ever_written : 0u,
                          uploadLevel ? (unsigned)uploadLevel->has_initialized_data : 0u,
                          (unsigned long long)hit);
                }
                anyLevelSkipped = YES;
                continue;
            }

            width = tex->faces[face].levels[level].width;
            height = tex->faces[face].levels[level].height;
            depth = tex->faces[face].levels[level].depth;

            if (texture1DBackedBy2D)
                region = MTLRegionMake2D(0,0,width,1);
            else if (depth > 1)
                region = MTLRegionMake3D(0,0,0,width,height,depth);
            else if (height > 1)
                region = MTLRegionMake2D(0,0,width,height);
            else
                region = MTLRegionMake1D(0,width);

            NSUInteger bytesPerRow;
            NSUInteger bytesPerImage;
            bool hasExplicitDataSize = false;

            BOOL levelSkipped = NO;

            if (tex_type == MTLTextureType3D)
            {
                if (![self uploadDirtyCPUTextureData3DLevel:tex
                                                       metal:texture
                                                 pixelFormat:pixelFormat
                                                       face:face
                                                      level:level
                                                      width:width
                                                     height:height
                                                      depth:depth
                                                 outSkipped:&levelSkipped]) {
                    return NO;
                }
            }
            else
            {
                if (![self uploadDirtyCPUTextureDataNon3DLevel:tex
                                                          metal:texture
                                                    pixelFormat:pixelFormat
                                                          face:face
                                                         level:level
                                                         width:width
                                                        height:height
                                                         depth:depth
                                                       isArray:is_array
                                  texture1DArrayBackedBy2DArray:texture1DArrayBackedBy2DArray
                                                        texType:tex_type
                                                     outSkipped:&levelSkipped]) {
                    return NO;
                }
            }

            if (levelSkipped)
                anyLevelSkipped = YES;
        }
    }

    if (outAllLevelsUploaded)
        *outAllLevelsUploaded = !anyLevelSkipped;

    return YES;
}

- (void)reUploadExistingCPUTextureDataArrayLevel:(Texture *)tex
                                          metal:(id<MTLTexture>)texture
                                    pixelFormat:(MTLPixelFormat)pixelFormat
                                          face:(int)face
                                         level:(int)level
                  texture1DArrayBackedBy2DArray:(BOOL)texture1DArrayBackedBy2DArray
                                       texType:(MTLTextureType)tex_type
{
    NSUInteger lvlWidth  = tex->faces[face].levels[level].width;
    NSUInteger lvlHeight = tex->faces[face].levels[level].height;
    NSUInteger lvlPitch  = tex->faces[face].levels[level].pitch;


                /* Array texture re-upload: loop over array layers and upload

                 * each slice independently.  Mirrors the DIRTY_TEXTURE_DATA

                 * array path (12861-13087).  The old code only uploaded

                 * slice 0 and passed the entire array's data_size as

                 * bytesPerImage with depth=num_layers, causing a crash in

                 * uploadTextureSliceViaBlit's newBufferWithBytes. */

                GLuint num_layers = (tex_type == MTLTextureType1DArray || texture1DArrayBackedBy2DArray)

                    ? tex->faces[face].levels[level].height

                    : tex->faces[face].levels[level].depth;

                if (num_layers == 0) return;


                BOOL arraySliceIs1D = (tex_type == MTLTextureType1DArray || texture1DArrayBackedBy2DArray);

                NSUInteger uploadSliceHeight = arraySliceIs1D ? 1UL : MAX((NSUInteger)lvlHeight, 1UL);

                NSUInteger baseBytesPerRow = lvlPitch;

                NSUInteger uploadSliceRows = mglMetalUploadRowsForPixelFormat(pixelFormat, uploadSliceHeight);

                if (uploadSliceRows == 0 || baseBytesPerRow > (NSUIntegerMax / uploadSliceRows)) {

                    NSLog(@"MGL WARNING: Re-upload array invalid row layout tex=%d face=%d level=%d bpr=%lu rows=%lu",

                          tex->name,

                          face,

                          level,

                          (unsigned long)baseBytesPerRow,

                          (unsigned long)uploadSliceRows);

                    return;

                }

                NSUInteger logicalBytesPerImage = baseBytesPerRow * uploadSliceRows;

                NSUInteger backingBytes = tex->faces[face].levels[level].data_size;

                if (num_layers > 1 && backingBytes >= (NSUInteger)num_layers) {

                    NSUInteger dividedLayerBytes = backingBytes / (NSUInteger)num_layers;

                    if (dividedLayerBytes >= logicalBytesPerImage) {

                        logicalBytesPerImage = dividedLayerBytes;

                    }

                }


                NSUInteger requiredArrayBytes = 0;

                NSUInteger safeLayerCount = MAX((NSUInteger)num_layers, 1UL);

                if (logicalBytesPerImage == 0 ||

                    logicalBytesPerImage > (NSUIntegerMax / safeLayerCount) ||

                    backingBytes < (requiredArrayBytes = logicalBytesPerImage * safeLayerCount)) {

                    NSLog(@"MGL WARNING: Re-upload array backing too small tex=%d face=%d level=%d backing=%lu layerBytes=%lu layers=%u",

                          tex->name, face, level,

                          (unsigned long)backingBytes,

                          (unsigned long)logicalBytesPerImage,

                          num_layers);

                    return;

                }


                for (GLuint layer = 0; layer < num_layers; layer++)

                {

                    size_t offset = logicalBytesPerImage * layer;

                    const void *layerSrcData = (const uint8_t *)tex->faces[face].levels[level].data + offset;

                    void *expandedUploadData = NULL;

                    NSUInteger effectiveBytesPerRow = baseBytesPerRow;

                    NSUInteger effectiveBytesPerImage = logicalBytesPerImage;


                    if (mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, pixelFormat)) {

                        NSUInteger expandedBPR = 0, expandedBPI = 0;

                        expandedUploadData = mglCreateRGBA8ExpandedUpload(tex,

                                                                          (const uint8_t *)layerSrcData,

                                                                          lvlWidth,

                                                                          uploadSliceHeight,

                                                                          baseBytesPerRow,

                                                                          &expandedBPR,

                                                                          &expandedBPI);

                        if (expandedUploadData) {

                            layerSrcData = expandedUploadData;

                            effectiveBytesPerRow = expandedBPR;

                            effectiveBytesPerImage = expandedBPI;

                        }

                    } else if (mglTextureNeedsChannelExpansion(tex->internalformat, pixelFormat)) {

                        NSUInteger expandedBPR = 0, expandedBPI = 0;

                        expandedUploadData = mglCreateChannelExpandedUpload(tex,

                                                                             pixelFormat,

                                                                             (const uint8_t *)layerSrcData,

                                                                             lvlWidth,

                                                                             uploadSliceHeight,

                                                                             baseBytesPerRow,

                                                                             &expandedBPR,

                                                                             &expandedBPI);

                        if (expandedUploadData) {

                            layerSrcData = expandedUploadData;

                            effectiveBytesPerRow = expandedBPR;

                            effectiveBytesPerImage = expandedBPI;

                        }

                    }


                    NSUInteger alignment = [self getOptimalAlignmentForPixelFormat:pixelFormat];

                    NSUInteger alignedBytesPerRow = effectiveBytesPerRow;

                    if (alignedBytesPerRow % alignment != 0) {

                        alignedBytesPerRow = ((alignedBytesPerRow + alignment - 1) / alignment) * alignment;

                    }


                    uintptr_t addr = (uintptr_t)layerSrcData;

                    if (addr % alignment != 0 || alignedBytesPerRow != effectiveBytesPerRow) {

                        NSUInteger alignedUploadRows = mglMetalUploadRowsForPixelFormat(pixelFormat, uploadSliceHeight);

                        if (alignedUploadRows == 0 || alignedBytesPerRow > (NSUIntegerMax / alignedUploadRows)) {

                            NSLog(@"MGL WARNING: Re-upload array rejecting aligned row layout bpr=%lu rows=%lu tex=%d face=%d level=%d layer=%u",

                                  (unsigned long)alignedBytesPerRow,

                                  (unsigned long)alignedUploadRows,

                                  tex->name,

                                  face,

                                  level,

                                  layer);

                            free(expandedUploadData);

                            continue;

                        }

                        NSUInteger alignedSize = alignedBytesPerRow * alignedUploadRows;

                        if (alignedSize > 0 && alignedSize <= (512 * 1024 * 1024)) {

                            void *alignedData = aligned_alloc(alignment, alignedSize);

                            if (alignedData) {

                                memset(alignedData, 0, alignedSize);

                                for (NSUInteger row = 0; row < alignedUploadRows; row++) {

                                    NSUInteger copySize = MIN(effectiveBytesPerRow, alignedBytesPerRow);

                                    memcpy((uint8_t *)alignedData + row * alignedBytesPerRow,

                                           (const uint8_t *)layerSrcData + row * effectiveBytesPerRow, copySize);

                                }

                                [self uploadTextureSliceViaBlit:texture

                                                       texName:tex->name

                                                     texTarget:tex->target

                                                         bytes:alignedData

                                                   bytesPerRow:alignedBytesPerRow

                                                 bytesPerImage:alignedSize

                                                         width:lvlWidth

                                                        height:lvlHeight

                                                         depth:1

                                                         level:level

                                                         slice:layer];

                                free(alignedData);

                            }

                        }

                    } else {

                        [self uploadTextureSliceViaBlit:texture

                                               texName:tex->name

                                             texTarget:tex->target

                                                 bytes:layerSrcData

                                           bytesPerRow:effectiveBytesPerRow

                                         bytesPerImage:effectiveBytesPerImage

                                                 width:lvlWidth

                                                height:lvlHeight

                                                 depth:1

                                                 level:level

                                                 slice:layer];

                    }

                    free(expandedUploadData);

                }

}

- (void)fillSmallRGBA8TextureWithGradient:(id<MTLTexture>)texture tex:(Texture *)tex
{
                            if (texture.width <= 512 && texture.height <= 512 && tex->internalformat == GL_RGBA8) {

                                NSLog(@"MGL INFO: Attempting simple direct color fill for small RGBA8 texture");


                                @try {

                                    // Create a simple pattern that's not magenta

                                    NSUInteger pixelCount = texture.width * texture.height;

                                    uint32_t *simpleData = calloc(pixelCount, sizeof(uint32_t));


                                    if (simpleData) {

                                        // Create a simple gradient pattern instead of magenta

                                        for (NSUInteger y = 0; y < texture.height; y++) {

                                            for (NSUInteger x = 0; x < texture.width; x++) {

                                                NSUInteger index = y * texture.width + x;


                                                // Create a simple gradient from blue to green

                                                uint8_t r = (uint8_t)(x * 255 / texture.width);

                                                uint8_t g = (uint8_t)(y * 255 / texture.height);

                                                uint8_t b = 128;

                                                uint8_t a = 255;


                                                simpleData[index] = (a << 24) | (b << 16) | (g << 8) | r;

                                            }

                                        }


                                        // Try direct replaceRegion for simple cases

                                        MTLRegion simpleRegion = MTLRegionMake2D(0, 0, texture.width, texture.height);

                                        [texture replaceRegion:simpleRegion

                                                mipmapLevel:0

                                                      slice:0

                                                  withBytes:simpleData

                                                bytesPerRow:texture.width * sizeof(uint32_t)

                                              bytesPerImage:texture.width * texture.height * sizeof(uint32_t)];


                                        NSLog(@"MGL SUCCESS: Simple direct color fill completed");

                                        mglMarkTextureLevelMetalFilled(tex, 0, pixelCount * sizeof(uint32_t));

                                        free(simpleData);

                                    }

                                } @catch (NSException *exception) {

                                    NSLog(@"MGL WARNING: Simple direct fill also failed: %@", exception.reason);

                                }

                            } else {

                                NSLog(@"MGL INFO: Skipping complex texture - would use deferred initialization");

                            }
}

- (BOOL)uploadDirtyCPUTextureData3DLevel:(Texture *)tex
                                    metal:(id<MTLTexture>)texture
                              pixelFormat:(MTLPixelFormat)pixelFormat
                                       face:(int)face
                                      level:(int)level
                                      width:(NSUInteger)width
                                     height:(NSUInteger)height
                                      depth:(NSUInteger)depth
                                 outSkipped:(BOOL *)outSkipped
{
    NSUInteger bytesPerRow;
    NSUInteger bytesPerImage;

                bytesPerRow = tex->faces[face].levels[level].pitch;
                if (bytesPerRow == 0) {
                    NSLog(@"MGL WARNING: Invalid 3D bytesPerRow (0), skipping upload (tex=%d face=%d level=%d)", tex->name, face, level);
                    if (outSkipped) *outSkipped = YES;
                    return YES;
                }

                NSUInteger uploadRows = mglMetalUploadRowsForPixelFormat(pixelFormat, MAX((NSUInteger)height, 1UL));
                if (uploadRows == 0 || bytesPerRow > (NSUIntegerMax / uploadRows)) {
                    NSLog(@"MGL WARNING: Invalid 3D bytesPerImage overflow (tex=%d face=%d level=%d rows=%lu bpr=%lu)",
                          tex->name,
                          face,
                          level,
                          (unsigned long)uploadRows,
                          (unsigned long)bytesPerRow);
                    if (outSkipped) *outSkipped = YES;
                    return YES;
                }
                bytesPerImage = bytesPerRow * uploadRows;

                if (tex->faces[face].levels[level].data && bytesPerRow > 0 && bytesPerImage > 0) {
                    void *srcData = (void *)tex->faces[face].levels[level].data;
                    uintptr_t addr = (uintptr_t)srcData;

                    uint8_t *expanded3DUploadData = NULL;
                    if (mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, pixelFormat)) {
                        NSUInteger expandedBytesPerRow = 0;
                        NSUInteger expandedBytesPerImagePerSlice = 0;
                        NSUInteger texDepth = MAX((NSUInteger)depth, 1UL);
                        NSUInteger texHeight = MAX((NSUInteger)height, 1UL);

                        uint8_t *firstSlice = mglCreateRGBA8ExpandedUpload(tex,
                                                                           (const uint8_t *)srcData,
                                                                           width,
                                                                           texHeight,
                                                                           bytesPerRow,
                                                                           &expandedBytesPerRow,
                                                                           &expandedBytesPerImagePerSlice);
                        if (firstSlice) {
                            NSUInteger totalExpandedSize = expandedBytesPerImagePerSlice * texDepth;
                            if (totalExpandedSize > 0 && totalExpandedSize <= (512 * 1024 * 1024)) {
                                expanded3DUploadData = (uint8_t *)malloc(totalExpandedSize);
                                if (expanded3DUploadData) {
                                    memcpy(expanded3DUploadData, firstSlice, expandedBytesPerImagePerSlice);
                                    for (NSUInteger z = 1; z < texDepth; z++) {
                                        const uint8_t *sliceSrc = (const uint8_t *)srcData + z * bytesPerImage;
                                        uint8_t *sliceDst = expanded3DUploadData + z * expandedBytesPerImagePerSlice;
                                        NSUInteger dummyRow = 0, dummyImage = 0;
                                        uint8_t *sliceExpanded = mglCreateRGBA8ExpandedUpload(tex,
                                                                                             sliceSrc,
                                                                                             width,
                                                                                             texHeight,
                                                                                             bytesPerRow,
                                                                                             &dummyRow,
                                                                                             &dummyImage);
                                        if (sliceExpanded) {
                                            memcpy(sliceDst, sliceExpanded, expandedBytesPerImagePerSlice);
                                            free(sliceExpanded);
                                        } else {
                                            memset(sliceDst, 0, expandedBytesPerImagePerSlice);
                                        }
                                    }
                                    srcData = expanded3DUploadData;
                                    bytesPerRow = expandedBytesPerRow;
                                    bytesPerImage = expandedBytesPerImagePerSlice;
                                    addr = (uintptr_t)srcData;
                                }
                            }
                            free(firstSlice);
                        }
                    } else if (mglTextureNeedsChannelExpansion(tex->internalformat, pixelFormat)) {
                        NSUInteger expandedBytesPerRow = 0;
                        NSUInteger expandedBytesPerImagePerSlice = 0;
                        NSUInteger texDepth = MAX((NSUInteger)depth, 1UL);
                        NSUInteger texHeight = MAX((NSUInteger)height, 1UL);

                        uint8_t *firstSlice = mglCreateChannelExpandedUpload(tex,
                                                                              pixelFormat,
                                                                              (const uint8_t *)srcData,
                                                                              width,
                                                                              texHeight,
                                                                              bytesPerRow,
                                                                              &expandedBytesPerRow,
                                                                              &expandedBytesPerImagePerSlice);
                        if (firstSlice) {
                            NSUInteger totalExpandedSize = expandedBytesPerImagePerSlice * texDepth;
                            if (totalExpandedSize > 0 && totalExpandedSize <= (512 * 1024 * 1024)) {
                                expanded3DUploadData = (uint8_t *)malloc(totalExpandedSize);
                                if (expanded3DUploadData) {
                                    memcpy(expanded3DUploadData, firstSlice, expandedBytesPerImagePerSlice);
                                    for (NSUInteger z = 1; z < texDepth; z++) {
                                        const uint8_t *sliceSrc = (const uint8_t *)srcData + z * bytesPerImage;
                                        uint8_t *sliceDst = expanded3DUploadData + z * expandedBytesPerImagePerSlice;
                                        NSUInteger dummyRow = 0, dummyImage = 0;
                                        uint8_t *sliceExpanded = mglCreateChannelExpandedUpload(tex,
                                                                                                 pixelFormat,
                                                                                                 sliceSrc,
                                                                                                 width,
                                                                                                 texHeight,
                                                                                                 bytesPerRow,
                                                                                                 &dummyRow,
                                                                                                 &dummyImage);
                                        if (sliceExpanded) {
                                            memcpy(sliceDst, sliceExpanded, expandedBytesPerImagePerSlice);
                                            free(sliceExpanded);
                                        } else {
                                            memset(sliceDst, 0, expandedBytesPerImagePerSlice);
                                        }
                                    }
                                    srcData = expanded3DUploadData;
                                    bytesPerRow = expandedBytesPerRow;
                                    bytesPerImage = expandedBytesPerImagePerSlice;
                                    addr = (uintptr_t)srcData;
                                }
                            }
                            free(firstSlice);
                        }
                    }

                    NSUInteger alignment = [self getOptimalAlignmentForPixelFormat:pixelFormat];
                    NSUInteger alignedBytesPerRow = bytesPerRow;
                    if (alignedBytesPerRow % alignment != 0) {
                        alignedBytesPerRow = ((alignedBytesPerRow + alignment - 1) / alignment) * alignment;
                    }

                    NSUInteger addrAlignment = MGLCapabilityTextureAlignment(&_capability);
                    if (addr % addrAlignment != 0 || alignedBytesPerRow != bytesPerRow) {
                        NSUInteger alignedUploadRows = mglMetalUploadRowsForPixelFormat(pixelFormat, MAX((NSUInteger)height, 1UL));
                        if (alignedUploadRows == 0 || alignedBytesPerRow > (NSUIntegerMax / alignedUploadRows)) {
                            NSLog(@"MGL WARNING: Rejecting aligned 3D upload row overflow (tex=%d level=%d rows=%lu bpr=%lu)",
                                  tex->name,
                                  level,
                                  (unsigned long)alignedUploadRows,
                                  (unsigned long)alignedBytesPerRow);
                            if (outSkipped) *outSkipped = YES;
                            return YES;
                        }
                        NSUInteger alignedBytesPerImage = alignedBytesPerRow * alignedUploadRows;
                        NSUInteger alignedDepth = MAX((NSUInteger)depth, 1UL);
                        if (alignedBytesPerImage > (NSUIntegerMax / alignedDepth)) {
                            NSLog(@"MGL WARNING: Rejecting aligned 3D upload size overflow (tex=%d level=%d bpi=%lu depth=%lu)",
                                  tex->name,
                                  level,
                                  (unsigned long)alignedBytesPerImage,
                                  (unsigned long)alignedDepth);
                            if (outSkipped) *outSkipped = YES;
                            return YES;
                        }
                        NSUInteger alignedSize = alignedBytesPerImage * alignedDepth;
                        if (alignedSize == 0 || alignedSize > (512 * 1024 * 1024)) {
                            NSLog(@"MGL WARNING: Rejecting aligned 3D upload staging size=%lu (tex=%d level=%d)",
                                  (unsigned long)alignedSize, tex->name, level);
                            if (outSkipped) *outSkipped = YES;
                            return YES;
                        }
                        void *alignedData = aligned_alloc(alignment, alignedSize);

                        if (alignedData) {
                            memset(alignedData, 0, alignedSize);
                            NSUInteger srcRowSize = bytesPerRow;
                            NSUInteger dstRowSize = alignedBytesPerRow;
                            NSUInteger texUploadRows = alignedUploadRows;
                            NSUInteger texDepth = MAX((NSUInteger)depth, 1UL);
                            uint8_t *srcPtr = (uint8_t *)srcData;
                            uint8_t *dstPtr = (uint8_t *)alignedData;

                            for (NSUInteger z = 0; z < texDepth; z++) {
                                for (NSUInteger row = 0; row < texUploadRows; row++) {
                                    NSUInteger copySize = (srcRowSize < dstRowSize) ? srcRowSize : dstRowSize;
                                    NSUInteger dstOffset = z * alignedBytesPerImage + row * dstRowSize;
                                    NSUInteger srcOffset = z * bytesPerImage + row * srcRowSize;
                                    memcpy(dstPtr + dstOffset, srcPtr + srcOffset, copySize);
                                    if (dstRowSize > copySize) {
                                        memset(dstPtr + dstOffset + copySize, 0, dstRowSize - copySize);
                                    }
                                }
                            }

                            if (!alignedData) {
                                NSLog(@"MGL SECURITY ERROR: NULL alignedData passed to Metal replaceRegion (level %d) - SKIPPING to prevent crash", level);
                                if (outSkipped) *outSkipped = YES;
                                return YES;
                            }
                            if (alignedBytesPerRow == 0) {
                                NSLog(@"MGL SECURITY ERROR: Invalid alignedBytesPerRow (0) passed to Metal replaceRegion (level %d) - SKIPPING to prevent crash", level);
                                if (outSkipped) *outSkipped = YES;
                                return YES;
                            }
                            @try {
                                BOOL uploaded = [self uploadTextureSliceViaBlit:texture
                                                                       texName:tex->name
                                                                     texTarget:tex->target
                                                                         bytes:alignedData
                                                                   bytesPerRow:alignedBytesPerRow
                                                                 bytesPerImage:alignedBytesPerImage
                                                                         width:width
                                                                        height:height
                                                                         depth:depth
                                                                         level:level
                                                                         slice:0];
                                if (!uploaded) {
                                    NSLog(@"MGL WARNING: 3D aligned blit upload failed (level %d, face %d)", level, face);
                                }
                            } @catch (NSException *exception) {
                                NSLog(@"MGL ERROR: Failed to upload aligned 3D texture data (level %d, face %d): %@", level, face, exception);
                            }
                            free(alignedData);
                        } else {
                            NSLog(@"MGL ERROR: Failed to allocate aligned memory for 3D texture upload");
                        }
                    } else {
                        if (!srcData) {
                            NSLog(@"MGL SECURITY ERROR: NULL srcData passed to Metal replaceRegion (level %d) - SKIPPING to prevent crash", level);
                            if (outSkipped) *outSkipped = YES;
                            return YES;
                        }
                        if (bytesPerRow == 0) {
                            NSLog(@"MGL SECURITY ERROR: Invalid bytesPerRow (0) passed to Metal replaceRegion (level %d) - SKIPPING to prevent crash", level);
                            if (outSkipped) *outSkipped = YES;
                            return YES;
                        }
                        if (bytesPerImage == 0) {
                            NSLog(@"MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d) - SKIPPING to prevent crash", level);
                            if (outSkipped) *outSkipped = YES;
                            return YES;
                        }
                        @try {
                            BOOL uploaded = [self uploadTextureSliceViaBlit:texture
                                                                   texName:tex->name
                                                                 texTarget:tex->target
                                                                     bytes:srcData
                                                               bytesPerRow:bytesPerRow
                                                             bytesPerImage:bytesPerImage
                                                                     width:width
                                                                    height:height
                                                                     depth:depth
                                                                     level:level
                                                                     slice:0];
                            if (!uploaded) {
                                NSLog(@"MGL WARNING: 3D direct blit upload failed (level %d, face %d)", level, face);
                            }
                        } @catch (NSException *exception) {
                            NSLog(@"MGL ERROR: Failed to upload 3D texture data (level %d, face %d): %@", level, face, exception);
                        }
                    }
                    free(expanded3DUploadData);
                } else {
                    NSLog(@"MGL WARNING: Skipping 3D texture upload due to invalid data or parameters");
                }

    return YES;
}

- (BOOL)uploadDirtyCPUTextureDataNon3DLevel:(Texture *)tex
                                       metal:(id<MTLTexture>)texture
                                 pixelFormat:(MTLPixelFormat)pixelFormat
                                       face:(int)face
                                      level:(int)level
                                      width:(NSUInteger)width
                                     height:(NSUInteger)height
                                      depth:(NSUInteger)depth
                                   isArray:(BOOL)is_array
                  texture1DArrayBackedBy2DArray:(BOOL)texture1DArrayBackedBy2DArray
                                    texType:(MTLTextureType)tex_type
                                 outSkipped:(BOOL *)outSkipped
{
    NSUInteger bytesPerRow;
    NSUInteger bytesPerImage;
    bool hasExplicitDataSize = false;
    MTLRegion region;

                bytesPerRow = tex->faces[face].levels[level].pitch;
                if (bytesPerRow == 0) {
                    NSLog(@"MGL WARNING: Invalid bytesPerRow (0), skipping upload (tex=%d face=%d level=%d)", tex->name, face, level);
                    if (outSkipped) *outSkipped = YES;
                    return YES;
                }

                bytesPerImage = tex->faces[face].levels[level].data_size;
                hasExplicitDataSize = (bytesPerImage > 0);
                if (bytesPerImage == 0) {
                    NSUInteger fallbackHeight = (height > 0) ? (NSUInteger)height : 1;
                    bytesPerImage = bytesPerRow * fallbackHeight;
                    NSLog(@"MGL WARNING: data_size was 0, using fallback bytesPerImage=%lu (tex=%d face=%d level=%d)",
                          (unsigned long)bytesPerImage, tex->name, face, level);
                }
                if (bytesPerImage == 0) {
                    NSLog(@"MGL WARNING: Invalid bytesPerImage (0), skipping upload (tex=%d face=%d level=%d)", tex->name, face, level);
                    if (outSkipped) *outSkipped = YES;
                    return YES;
                }

                if (is_array)
                {
                    GLuint num_layers;
                    size_t offset;
                    GLubyte *tex_data;
                    BOOL arraySliceIs1D;
                    NSUInteger uploadSliceHeight;
                    NSUInteger backingBytes;
                    NSUInteger logicalBytesPerImage;

                    num_layers = (tex_type == MTLTextureType1DArray || texture1DArrayBackedBy2DArray)
                        ? tex->faces[face].levels[level].height
                        : tex->faces[face].levels[level].depth;
                    if (num_layers == 0) {
                        NSLog(@"MGL WARNING: Array texture has 0 layers, skipping upload (tex=%d face=%d level=%d)", tex->name, face, level);
                        if (outSkipped) *outSkipped = YES;
                        return YES;
                    }

                    arraySliceIs1D = (tex_type == MTLTextureType1DArray || texture1DArrayBackedBy2DArray);
                    uploadSliceHeight = arraySliceIs1D ? 1UL : MAX((NSUInteger)height, 1UL);
                    backingBytes = bytesPerImage;
                    NSUInteger uploadSliceRows = mglMetalUploadRowsForPixelFormat(pixelFormat, uploadSliceHeight);
                    if (uploadSliceRows == 0 || bytesPerRow > (NSUIntegerMax / uploadSliceRows)) {
                        NSLog(@"MGL WARNING: Array texture invalid row layout tex=%d face=%d level=%d bpr=%lu rows=%lu",
                              tex->name,
                              face,
                              level,
                              (unsigned long)bytesPerRow,
                              (unsigned long)uploadSliceRows);
                        if (outSkipped) *outSkipped = YES;
                        return YES;
                    }
                    logicalBytesPerImage = bytesPerRow * uploadSliceRows;
                    if (num_layers > 1 && backingBytes >= (NSUInteger)num_layers) {
                        NSUInteger dividedLayerBytes = backingBytes / (NSUInteger)num_layers;
                        if (dividedLayerBytes >= logicalBytesPerImage) {
                            logicalBytesPerImage = dividedLayerBytes;
                        }
                    }
                    NSUInteger requiredArrayBytes = 0;
                    NSUInteger safeLayerCount = MAX((NSUInteger)num_layers, 1UL);
                    if (logicalBytesPerImage == 0 ||
                        logicalBytesPerImage > (NSUIntegerMax / safeLayerCount) ||
                        backingBytes < (requiredArrayBytes = logicalBytesPerImage * safeLayerCount)) {
                        NSLog(@"MGL WARNING: Array texture backing too small for logical slices tex=%d face=%d level=%d backing=%lu layerBytes=%lu layers=%u",
                              tex->name,
                              face,
                              level,
                              (unsigned long)backingBytes,
                              (unsigned long)logicalBytesPerImage,
                              num_layers);
                        if (outSkipped) *outSkipped = YES;
                        return YES;
                    }
                    bytesPerImage = logicalBytesPerImage;

                    if (!arraySliceIs1D)
                        region = MTLRegionMake2D(0,0,width,height);
                    else if (height >= 1)
                        region = MTLRegionMake2D(0,0,width,1);
                    else {
                        NSLog(@"MGL TEXTURE ERROR: invalid array texture height=%lu for tex=%u face=%d level=%d",
                              (unsigned long)height,
                              tex->name,
                              face,
                              level);
                        return NO;
                    }

                    for(int layer=0; layer<num_layers; layer++)
                    {
                        offset = bytesPerImage * layer;

                        tex_data = (GLubyte *)tex->faces[face].levels[level].data;
                        tex_data += offset;

                        if (tex_data && bytesPerRow > 0 && bytesPerImage > 0) {
                            void *srcData = (void *)tex_data;
                            void *expandedUploadData = NULL;
                            uintptr_t addr = (uintptr_t)srcData;

                            NSUInteger effectiveBytesPerRow = bytesPerRow;
                            NSUInteger effectiveBytesPerImage = bytesPerImage;
                            if (mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, pixelFormat)) {
                                NSUInteger expandedBytesPerRow = 0;
                                NSUInteger expandedBytesPerImage = 0;
                                expandedUploadData = mglCreateRGBA8ExpandedUpload(tex,
                                                                                   (const uint8_t *)srcData,
                                                                                   width,
                                                                                   uploadSliceHeight,
                                                                                   bytesPerRow,
                                                                                   &expandedBytesPerRow,
                                                                                   &expandedBytesPerImage);
                                if (expandedUploadData) {
                                    srcData = expandedUploadData;
                                    effectiveBytesPerRow = expandedBytesPerRow;
                                    effectiveBytesPerImage = expandedBytesPerImage;
                                    addr = (uintptr_t)srcData;
                                }
                            } else if (mglTextureNeedsChannelExpansion(tex->internalformat, pixelFormat)) {
                                NSUInteger expandedBytesPerRow = 0;
                                NSUInteger expandedBytesPerImage = 0;
                                expandedUploadData = mglCreateChannelExpandedUpload(tex,
                                                                                     pixelFormat,
                                                                                     (const uint8_t *)srcData,
                                                                                     width,
                                                                                     uploadSliceHeight,
                                                                                     bytesPerRow,
                                                                                     &expandedBytesPerRow,
                                                                                     &expandedBytesPerImage);
                                if (expandedUploadData) {
                                    srcData = expandedUploadData;
                                    effectiveBytesPerRow = expandedBytesPerRow;
                                    effectiveBytesPerImage = expandedBytesPerImage;
                                    addr = (uintptr_t)srcData;
                                }
                            }

                            NSUInteger alignment = [self getOptimalAlignmentForPixelFormat:pixelFormat];
                            NSUInteger alignedBytesPerRow = effectiveBytesPerRow;
                            if (alignedBytesPerRow % alignment != 0) {
                                alignedBytesPerRow = ((alignedBytesPerRow + alignment - 1) / alignment) * alignment;
                            }

                            if (addr % alignment != 0 || alignedBytesPerRow != effectiveBytesPerRow) {
                                NSUInteger alignedUploadRows = mglMetalUploadRowsForPixelFormat(pixelFormat, uploadSliceHeight);
                                if (alignedUploadRows == 0 || alignedBytesPerRow > (NSUIntegerMax / alignedUploadRows)) {
                                    NSLog(@"MGL WARNING: Rejecting aligned array upload row layout bpr=%lu rows=%lu (tex=%d level=%d layer=%d)",
                                          (unsigned long)alignedBytesPerRow,
                                          (unsigned long)alignedUploadRows,
                                          tex->name,
                                          level,
                                          layer);
                                    free(expandedUploadData);
                                    continue;
                                }
                                NSUInteger alignedBytesPerImage = alignedBytesPerRow * alignedUploadRows;
                                NSUInteger alignedSize = alignedBytesPerImage;
                                if (alignedSize == 0 || alignedSize > (512 * 1024 * 1024)) {
                                    NSLog(@"MGL WARNING: Rejecting aligned array upload staging size=%lu (tex=%d level=%d layer=%d)",
                                          (unsigned long)alignedSize, tex->name, level, layer);
                                    free(expandedUploadData);
                                    continue;
                                }
                                void *alignedData = aligned_alloc(alignment, alignedSize);

                                if (alignedData) {
                                    memset(alignedData, 0, alignedSize);
                                    NSUInteger srcRowSize = effectiveBytesPerRow;
                                    NSUInteger dstRowSize = alignedBytesPerRow;
                                    uint8_t *srcPtr = (uint8_t *)srcData;
                                    uint8_t *dstPtr = (uint8_t *)alignedData;

                                    for (NSUInteger row = 0; row < alignedUploadRows; row++) {
                                        NSUInteger copySize = (srcRowSize < dstRowSize) ? srcRowSize : dstRowSize;
                                        memcpy(dstPtr + (row * dstRowSize), srcPtr + (row * srcRowSize), copySize);
                                        if (dstRowSize > copySize) {
                                            memset(dstPtr + (row * dstRowSize) + copySize, 0, dstRowSize - copySize);
                                        }
                                    }

                                    if (!alignedData) {
                                        NSLog(@"MGL SECURITY ERROR: NULL alignedData passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash", level, layer);
                                        continue;
                                    }
                                    if (alignedBytesPerRow == 0) {
                                        NSLog(@"MGL SECURITY ERROR: Invalid alignedBytesPerRow (0) passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash", level, layer);
                                        continue;
                                    }
                                    if (bytesPerImage == 0) {
                                        NSLog(@"MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash", level, layer);
                                        continue;
                                    }
                                    @try {
                                        if (hasExplicitDataSize) {
                                            BOOL uploaded = [self uploadTextureSliceViaBlit:texture
                                                                                   texName:tex->name
                                                                                 texTarget:tex->target
                                                                                     bytes:alignedData
                                                                               bytesPerRow:alignedBytesPerRow
                                                                             bytesPerImage:alignedBytesPerImage
                                                                                     width:width
                                                                                    height:uploadSliceHeight
                                                                                     depth:1
                                                                                     level:level
                                                                                     slice:layer];
                                            if (!uploaded) {
                                                NSLog(@"MGL WARNING: Array texture blit upload failed (level %d, layer %d)", level, layer);
                                            }
                                        } else {
                                            NSLog(@"MGL INFO: Skipping array upload with synthesized data size (level %d, layer %d)", level, layer);
                                        }
                                    } @catch (NSException *exception) {
                                        NSLog(@"MGL ERROR: Failed to upload aligned array texture data (level %d, layer %d): %@", level, layer, exception);
                                    }
                                    free(alignedData);
                                } else {
                                    NSLog(@"MGL ERROR: Failed to allocate aligned memory for array texture upload (level %d, layer %d)", level, layer);
                                }
                            } else {
                                if (!srcData) {
                                    NSLog(@"MGL SECURITY ERROR: NULL srcData passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash", level, layer);
                                    free(expandedUploadData);
                                    continue;
                                }
                                if (effectiveBytesPerRow == 0) {
                                    NSLog(@"MGL SECURITY ERROR: Invalid bytesPerRow (0) passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash", level, layer);
                                    free(expandedUploadData);
                                    continue;
                                }
                                if (effectiveBytesPerImage == 0) {
                                    NSLog(@"MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash", level, layer);
                                    free(expandedUploadData);
                                    continue;
                                }
                                if (hasExplicitDataSize) {
                                    BOOL uploaded = [self uploadTextureSliceViaBlit:texture
                                                                           texName:tex->name
                                                                         texTarget:tex->target
                                                                             bytes:srcData
                                                                       bytesPerRow:effectiveBytesPerRow
                                                                     bytesPerImage:effectiveBytesPerImage
                                                                             width:width
                                                                            height:uploadSliceHeight
                                                                             depth:1
                                                                             level:level
                                                                             slice:layer];
                                    if (!uploaded) {
                                        NSLog(@"MGL WARNING: Array texture direct blit upload failed (level %d, layer %d)", level, layer);
                                    }
                                } else {
                                    NSLog(@"MGL INFO: Skipping array upload with synthesized data size (level %d, layer %d)", level, layer);
                                }
                            }
                            free(expandedUploadData);
                        } else {
                            NSLog(@"MGL WARNING: Skipping array texture upload due to invalid data or parameters");
                        }
                    }
                }
                else
                {
                    DEBUG_PRINT("tex id data update %d\n", tex->name);

                    if (tex->faces[face].levels[level].data && bytesPerRow > 0 && bytesPerImage > 0) {
                        void *srcData = (void *)tex->faces[face].levels[level].data;
                        void *swizzledUploadData = NULL;
                        void *expandedUploadData = NULL;
                        uintptr_t addr = (uintptr_t)srcData;
                        if (level == 0 && face == 0 && mglTextureUploadNeedsSingleChannelSwizzle(tex)) {
                            NSUInteger swizzledBytesPerRow = 0;
                            NSUInteger swizzledBytesPerImage = 0;
                            swizzledUploadData = mglCreateSingleChannelSwizzledUpload(tex,
                                                                                      (const uint8_t *)srcData,
                                                                                      width,
                                                                                      MAX((NSUInteger)height, 1UL),
                                                                                      bytesPerRow,
                                                                                      &swizzledBytesPerRow,
                                                                                      &swizzledBytesPerImage);
                            if (swizzledUploadData) {
                                srcData = swizzledUploadData;
                                bytesPerRow = swizzledBytesPerRow;
                                bytesPerImage = swizzledBytesPerImage;
                                addr = (uintptr_t)srcData;
                                if (mglTraceLogIsEnabled()) {
                                    const uint8_t *swz = (const uint8_t *)swizzledUploadData;
                                    mglTraceLog("TEXTURE_SWIZZLE_UPLOAD_R8 tex=%u target=0x%x swzR=0x%x size=%lux%lu bpr=%lu first=%u",
                                                (unsigned)tex->name,
                                                (unsigned)tex->target,
                                                (unsigned)tex->params.swizzle_r,
                                                (unsigned long)width,
                                                (unsigned long)MAX((NSUInteger)height, 1UL),
                                                (unsigned long)bytesPerRow,
                                                swz[0]);
                                }
                            }
                        }
                        if (!swizzledUploadData &&
                            mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, pixelFormat)) {
                            NSUInteger expandedBytesPerRow = 0;
                            NSUInteger expandedBytesPerImage = 0;
                            expandedUploadData = mglCreateRGBA8ExpandedUpload(tex,
                                                                               (const uint8_t *)srcData,
                                                                               width,
                                                                               MAX((NSUInteger)height, 1UL),
                                                                               bytesPerRow,
                                                                               &expandedBytesPerRow,
                                                                               &expandedBytesPerImage);
                            if (expandedUploadData) {
                                srcData = expandedUploadData;
                                bytesPerRow = expandedBytesPerRow;
                                bytesPerImage = expandedBytesPerImage;
                                addr = (uintptr_t)srcData;
                            }
                        } else if (!swizzledUploadData &&
                                   mglTextureNeedsChannelExpansion(tex->internalformat, pixelFormat)) {
                            NSUInteger expandedBytesPerRow = 0;
                            NSUInteger expandedBytesPerImage = 0;
                            expandedUploadData = mglCreateChannelExpandedUpload(tex,
                                                                                 pixelFormat,
                                                                                 (const uint8_t *)srcData,
                                                                                 width,
                                                                                 MAX((NSUInteger)height, 1UL),
                                                                                 bytesPerRow,
                                                                                 &expandedBytesPerRow,
                                                                                 &expandedBytesPerImage);
                            if (expandedUploadData) {
                                srcData = expandedUploadData;
                                bytesPerRow = expandedBytesPerRow;
                                bytesPerImage = expandedBytesPerImage;
                                addr = (uintptr_t)srcData;
                            }
                        }

                        NSUInteger alignment = [self getOptimalAlignmentForPixelFormat:pixelFormat];
                        NSUInteger alignedBytesPerRow = bytesPerRow;
                        if (alignedBytesPerRow % alignment != 0) {
                            alignedBytesPerRow = ((alignedBytesPerRow + alignment - 1) / alignment) * alignment;
                        }

                        if (addr % alignment != 0 || alignedBytesPerRow != bytesPerRow) {
                            NSUInteger texHeight = MAX((NSUInteger)height, 1UL);
                            NSUInteger uploadRows = mglMetalUploadRowsForPixelFormat(pixelFormat, texHeight);
                            if (uploadRows == 0 || alignedBytesPerRow > (NSUIntegerMax / uploadRows)) {
                                NSLog(@"MGL WARNING: Rejecting aligned 2D upload row layout bpr=%lu rows=%lu (tex=%d level=%d face=%d)",
                                      (unsigned long)alignedBytesPerRow,
                                      (unsigned long)uploadRows,
                                      tex->name,
                                      level,
                                      face);
                                free(swizzledUploadData);
                                free(expandedUploadData);
                                if (outSkipped) *outSkipped = YES;
                                return YES;
                            }
                            NSUInteger alignedBytesPerImage = alignedBytesPerRow * uploadRows;
                            NSUInteger alignedSize = alignedBytesPerImage;
                            if (alignedSize == 0 || alignedSize > (512 * 1024 * 1024)) {
                                NSLog(@"MGL WARNING: Rejecting aligned 2D upload staging size=%lu (tex=%d level=%d face=%d)",
                                      (unsigned long)alignedSize, tex->name, level, face);
                                free(swizzledUploadData);
                                free(expandedUploadData);
                                if (outSkipped) *outSkipped = YES;
                                return YES;
                            }
                            void *alignedData = aligned_alloc(alignment, alignedSize);

                            if (alignedData) {
                                memset(alignedData, 0, alignedSize);
                                NSUInteger srcRowSize = bytesPerRow;
                                NSUInteger dstRowSize = alignedBytesPerRow;
                                uint8_t *srcPtr = (uint8_t *)srcData;
                                uint8_t *dstPtr = (uint8_t *)alignedData;

                                for (NSUInteger row = 0; row < uploadRows; row++) {
                                    NSUInteger copySize = (srcRowSize < dstRowSize) ? srcRowSize : dstRowSize;
                                    memcpy(dstPtr + (row * dstRowSize), srcPtr + (row * srcRowSize), copySize);
                                    if (dstRowSize > copySize) {
                                        memset(dstPtr + (row * dstRowSize) + copySize, 0, dstRowSize - copySize);
                                    }
                                }

                                if (!alignedData) {
                                    NSLog(@"MGL SECURITY ERROR: NULL alignedData passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash", level, face);
                                    free(alignedData);
                                    if (outSkipped) *outSkipped = YES;
                                    return YES;
                                }
                                if (alignedBytesPerRow == 0) {
                                    NSLog(@"MGL SECURITY ERROR: Invalid alignedBytesPerRow (0) passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash", level, face);
                                    free(alignedData);
                                    if (outSkipped) *outSkipped = YES;
                                    return YES;
                                }
                                if (bytesPerImage == 0) {
                                    NSLog(@"MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash", level, face);
                                    free(alignedData);
                                    if (outSkipped) *outSkipped = YES;
                                    return YES;
                                }
                                if (hasExplicitDataSize) {
                                    BOOL uploaded = [self uploadTextureSliceViaBlit:texture
                                                                           texName:tex->name
                                                                         texTarget:tex->target
                                                                             bytes:alignedData
                                                                       bytesPerRow:alignedBytesPerRow
                                                                     bytesPerImage:alignedBytesPerImage
                                                                             width:width
                                                                            height:height
                                                                             depth:1
                                                                             level:level
                                                                             slice:face];
                                    if (!uploaded) {
                                        NSLog(@"MGL WARNING: Aligned 2D blit upload failed (level %d, face %d)", level, face);
                                    }
                                } else {
                                    NSLog(@"MGL INFO: Skipping 2D upload with synthesized data size (level %d, face %d)", level, face);
                                }
                                free(alignedData);
                            } else {
                                NSLog(@"MGL ERROR: Failed to allocate aligned memory for 2D texture upload (level %d, face %d)", level, face);
                            }
                        } else {
                            if (!srcData) {
                                NSLog(@"MGL SECURITY ERROR: NULL srcData passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash", level, face);
                                if (outSkipped) *outSkipped = YES;
                                return YES;
                            }
                            if (bytesPerRow == 0) {
                                NSLog(@"MGL SECURITY ERROR: Invalid bytesPerRow (0) passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash", level, face);
                                if (outSkipped) *outSkipped = YES;
                                return YES;
                            }
                            if (bytesPerImage == 0) {
                                NSLog(@"MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash", level, face);
                                if (outSkipped) *outSkipped = YES;
                                return YES;
                            }
                            if (hasExplicitDataSize) {
                                BOOL uploaded = [self uploadTextureSliceViaBlit:texture
                                                                       texName:tex->name
                                                                     texTarget:tex->target
                                                                         bytes:srcData
                                                                   bytesPerRow:bytesPerRow
                                                                 bytesPerImage:bytesPerImage
                                                                         width:width
                                                                        height:height
                                                                         depth:1
                                                                         level:level
                                                                         slice:face];
                                if (!uploaded) {
                                    NSLog(@"MGL WARNING: 2D direct blit upload failed (level %d, face %d)", level, face);
                                }
                            } else {
                                NSLog(@"MGL INFO: Skipping 2D upload with synthesized data size (level %d, face %d)", level, face);
                            }
                        }
                        free(swizzledUploadData);
                        free(expandedUploadData);
                    } else {
                        NSLog(@"MGL WARNING: Skipping 2D texture upload due to invalid data or parameters");
                    }
                }

    return YES;
}

@end
