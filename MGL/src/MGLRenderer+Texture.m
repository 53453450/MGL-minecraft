// MGLRenderer+Texture.m
// Texture upload/download Metal path methods extracted from MGLRenderer.m

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Texture_Private.h"
#include "mgl_env_flag.h"
#include "mgl_render_cpp_objc.h"

int mglRendererCallbackResource(void *runtime_context,
                                GLMContext glm_ctx,
                                const MGLRenderCppResourceCallbackArgs *args)
{
    MGLRenderer *renderer = (__bridge MGLRenderer *)runtime_context;
    if (!renderer || !glm_ctx || !args) return 0;

    MTLRegion region = MTLRegionMake2D((NSUInteger)args->x,
                                       (NSUInteger)args->y,
                                       (NSUInteger)args->width,
                                       (NSUInteger)args->height);
    switch (args->kind) {
        case MGL_RENDER_CPP_RESOURCE_CALLBACK_READ_DRAWABLE:
            [renderer mtlReadDrawable:glm_ctx
                           pixelBytes:args->pixel_bytes
                          bytesPerRow:args->bytes_per_row
                        bytesPerImage:args->bytes_per_image
                           fromRegion:region];
            return 1;
        case MGL_RENDER_CPP_RESOURCE_CALLBACK_READ_INTEGER_PIXELS:
            [renderer mtlReadIntegerPixels:glm_ctx
                                pixelBytes:args->pixel_bytes
                               bytesPerRow:args->bytes_per_row
                             bytesPerImage:args->bytes_per_image
                                fromRegion:region
                                    format:args->format
                                      type:args->type];
            return 1;
        case MGL_RENDER_CPP_RESOURCE_CALLBACK_READ_DEPTH_PIXELS:
            [renderer mtlReadDepthPixels:glm_ctx
                              pixelBytes:args->pixel_bytes
                             bytesPerRow:args->bytes_per_row
                           bytesPerImage:args->bytes_per_image
                              fromRegion:region];
            return 1;
        case MGL_RENDER_CPP_RESOURCE_CALLBACK_GET_TEX_IMAGE:
            [renderer mtlGetTexImage:glm_ctx tex:args->texture
                          pixelBytes:args->pixel_bytes
                         bytesPerRow:args->bytes_per_row
                       bytesPerImage:args->bytes_per_image
                          fromRegion:region
                              format:args->format type:args->type
                         mipmapLevel:args->level slice:args->slice];
            return 1;
        case MGL_RENDER_CPP_RESOURCE_CALLBACK_GENERATE_MIPMAPS:
            [renderer mtlGenerateMipmaps:glm_ctx forTexture:args->texture];
            return 1;
        case MGL_RENDER_CPP_RESOURCE_CALLBACK_TEX_SUB_IMAGE:
            [renderer mtlTexSubImage:glm_ctx tex:args->texture buf:args->buffer
                          src_offset:args->source_offset
                           src_pitch:args->source_pitch
                      src_image_size:args->source_image_size
                           src_size:args->source_size
                              slice:args->slice level:args->level
                              width:args->width height:args->height
                              depth:args->depth
                            xoffset:args->x_offset yoffset:args->y_offset
                            zoffset:args->z_offset];
            return 1;
        case MGL_RENDER_CPP_RESOURCE_CALLBACK_TEX_SUB_IMAGE_BYTES:
            return [renderer mtlTexSubImageBytes:glm_ctx tex:args->texture
                                           bytes:args->bytes
                                       bytesSize:args->bytes_size
                                      src_offset:args->source_offset
                                       src_pitch:args->source_pitch
                                  src_image_size:args->source_image_size
                                           slice:args->slice level:args->level
                                           width:args->width height:args->height
                                           depth:args->depth
                                         xoffset:args->x_offset
                                         yoffset:args->y_offset
                                         zoffset:args->z_offset] ? 1 : 0;
        case MGL_RENDER_CPP_RESOURCE_CALLBACK_COPY_TEX_SUB_IMAGE:
            [renderer mtlCopyTexSubImage:glm_ctx tex:args->texture
                                   slice:args->slice
                             mipmapLevel:args->level
                                 xoffset:(NSInteger)args->x_offset
                                 yoffset:(NSInteger)args->y_offset
                                       x:args->x y:args->y
                                   width:args->width height:args->height];
            return 1;
        case MGL_RENDER_CPP_RESOURCE_CALLBACK_COPY_IMAGE_SUB_DATA:
            [renderer mtlCopyImageSubData:glm_ctx
                               srcTexture:args->source_texture
                                  srcLevel:args->source_level
                                      srcX:args->source_x
                                      srcY:args->source_y
                                      srcZ:args->source_z
                               dstTexture:args->destination_texture
                                  dstLevel:args->destination_level
                                      dstX:args->destination_x
                                      dstY:args->destination_y
                                      dstZ:args->destination_z
                                     width:(GLsizei)args->width
                                    height:(GLsizei)args->height
                                     depth:(GLsizei)args->depth];
            return 1;
        default:
            return 0;
    }
}

static MGLMetalBufferRef mglTextureCreateBuffer(MGLMetalDeviceRef device,
                                            NSUInteger length,
                                            MTLResourceOptions options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCppCreateBuffer(length, options, NULL, &buffer) == 0 &&
        buffer) {
        return (__bridge_transfer MGLMetalBufferRef)buffer;
    }
    return nil;
}

static MGLMetalBufferRef mglTextureCreateBufferWithBytes(
    MGLMetalDeviceRef device,
    const void *bytes,
    NSUInteger length,
    MTLResourceOptions options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCppCreateBufferWithBytes(bytes, length, options, NULL,
                                          &buffer) == 0 && buffer) {
        return (__bridge_transfer MGLMetalBufferRef)buffer;
    }
    return nil;
}

static MGLMetalTextureRef mglTextureCreateTexture(
    MGLMetalDeviceRef device,
    MTLTextureDescriptor *descriptor)
{
    (void)device;
    void *texture = NULL;
    MGLRenderCppTextureDescriptorState state =
        mglRenderCppTextureDescriptorStateFromObjC(descriptor);
    if (mglRenderCppCreateTextureFromState(&state, NULL, &texture) == 0 &&
        texture) {
        return (__bridge_transfer MGLMetalTextureRef)texture;
    }
    return nil;
}

static MGLMetalTextureRef mglTextureCreateBufferTexture(
    MGLMetalBufferRef buffer,
    MTLTextureDescriptor *descriptor,
    NSUInteger offset,
    NSUInteger bytesPerRow)
{
    void *texture = NULL;
    MGLRenderCppTextureDescriptorState state =
        mglRenderCppTextureDescriptorStateFromObjC(descriptor);
    if (mglRenderCppCreateBufferTextureFromState(
            (__bridge void *)buffer, &state, offset, bytesPerRow,
            &texture) == 0 && texture) {
        return (__bridge_transfer MGLMetalTextureRef)texture;
    }
    return nil;
}

static void mglTextureReplaceRegion(MGLMetalTextureRef texture,
                                    MTLRegion region,
                                    NSUInteger level,
                                    NSUInteger slice,
                                    const void *bytes,
                                    NSUInteger bytesPerRow,
                                    NSUInteger bytesPerImage,
                                    BOOL useSlice)
{
    if (mglRenderCppTextureReplaceRegion(
            (__bridge void *)texture,
            region.origin.x, region.origin.y, region.origin.z,
            region.size.width, region.size.height, region.size.depth,
            level, slice, bytes, bytesPerRow, bytesPerImage,
            useSlice ? 1 : 0) != 0) {
        [NSException raise:@"MGLTextureReplaceRegionError"
                    format:@"C++ texture replaceRegion failed (level=%lu slice=%lu)",
                           (unsigned long)level, (unsigned long)slice];
    }
}

static void mglTextureGetBytes(MGLMetalTextureRef texture,
                               void *bytes,
                               NSUInteger bytesPerRow,
                               NSUInteger bytesPerImage,
                               MTLRegion region,
                               NSUInteger level,
                               NSUInteger slice,
                               BOOL useSlice)
{
    if (mglRenderCppTextureGetBytes(
            (__bridge void *)texture, bytes, bytesPerRow, bytesPerImage,
            region.origin.x, region.origin.y, region.origin.z,
            region.size.width, region.size.height, region.size.depth,
            level, slice, useSlice ? 1 : 0) != 0) {
        [NSException raise:@"MGLTextureGetBytesError"
                    format:@"C++ texture getBytes failed (level=%lu slice=%lu)",
                           (unsigned long)level, (unsigned long)slice];
    }
}

static MGLMetalSamplerStateRef mglTextureCreateSampler(
    MGLMetalDeviceRef device,
    MTLSamplerDescriptor *descriptor)
{
    (void)device;
    void *sampler = NULL;
    if (mglRenderCppCreateSampler((__bridge void *)descriptor,
                                  &sampler) == 0 && sampler) {
        return (__bridge_transfer MGLMetalSamplerStateRef)sampler;
    }
    return nil;
}

static MGLMetalCommandBufferRef mglTextureCreateCommandBuffer(
    MGLMetalCommandQueueRef queue)
{
    if (!queue) return nil;
    void *commandBufferCPP = NULL;
    if (mglRenderCppCreateCommandBuffer((__bridge void *)queue,
                                         &commandBufferCPP) == 0 &&
        commandBufferCPP) {
        return (__bridge MGLMetalCommandBufferRef)commandBufferCPP;
    }
    return nil;
}

static MGLMetalBlitCommandEncoderRef mglTextureCreateBlitEncoder(
    MGLMetalCommandBufferRef commandBuffer)
{
    if (!commandBuffer) return nil;
    void *encoderCPP = NULL;
    if (mglRenderCppCreateBlitEncoder((__bridge void *)commandBuffer,
                                       &encoderCPP) == 0 && encoderCPP) {
        return (__bridge MGLMetalBlitCommandEncoderRef)encoderCPP;
    }
    return nil;
}

/* Owner-first adapter for work that is encoded on the renderer's current
 * command buffer. Dedicated command buffers continue to use the raw helper
 * above because they are not owned by MGLRenderPassManager. */
static MGLMetalBlitCommandEncoderRef mglTextureCreateCurrentBlitEncoder(
    void *commandBufferOwner)
{
    return mglRenderCreateBlitEncoderForCommandBufferOwner(
        commandBufferOwner);
}

static void mglTextureEndBlitEncoder(MGLMetalBlitCommandEncoderRef encoder)
{
    if (!encoder) return;
    (void)mglRenderCppEndBlitEncoder((__bridge void *)encoder);
}

static void mglTextureCommitCommandBuffer(MGLMetalCommandBufferRef commandBuffer)
{
    if (!commandBuffer) return;
    if (mglRenderCppCommitCommandBuffer(
            (__bridge void *)commandBuffer) != 0) {
        NSLog(@"MGL ERROR: Metal-cpp texture command-buffer commit failed");
    }
}

static void mglTextureWaitCommandBuffer(MGLMetalCommandBufferRef commandBuffer)
{
    if (!commandBuffer) return;
    if (mglRenderCppWaitCommandBuffer(
            (__bridge void *)commandBuffer) != 0) {
        NSLog(@"MGL ERROR: Metal-cpp texture command-buffer wait failed");
    }
}

static void mglTextureCopyTextureToBuffer(
    MGLMetalBlitCommandEncoderRef encoder,
    MGLMetalTextureRef source,
    NSUInteger sourceSlice,
    NSUInteger sourceLevel,
    MTLOrigin sourceOrigin,
    MTLSize sourceSize,
    MGLMetalBufferRef destination,
    NSUInteger destinationOffset,
    NSUInteger bytesPerRow,
    NSUInteger bytesPerImage)
{
    (void)mglRenderCppBlitCopyTextureToBuffer(
            (__bridge void *)encoder, (__bridge void *)source, sourceSlice,
            sourceLevel, sourceOrigin.x, sourceOrigin.y, sourceOrigin.z,
            sourceSize.width, sourceSize.height, sourceSize.depth,
            (__bridge void *)destination, destinationOffset, bytesPerRow,
            bytesPerImage);
}

@implementation MGLRenderer (Texture)

- (bool)copyTextureUploadWithDedicatedCommandBuffer:(MGLMetalBufferRef)sourceBuffer
                                        sourceOffset:(NSUInteger)sourceOffset
                                   sourceBytesPerRow:(NSUInteger)sourceBytesPerRow
                                 sourceBytesPerImage:(NSUInteger)sourceBytesPerImage
                                  sourceLayerStride:(NSUInteger)sourceLayerStride
                                          layerCount:(NSUInteger)layerCount
                                           sourceSize:(MTLSize)sourceSize
                                            toTexture:(MGLMetalTextureRef)texture
                                     destinationSlice:(NSUInteger)destinationSlice
                                     destinationLevel:(NSUInteger)destinationLevel
                                    destinationOrigin:(MTLOrigin)destinationOrigin
                                               reason:(const char *)reason
{
    MGL_ASSERT_GL_THREAD();
    if (!sourceBuffer || !texture || !_commandQueue || layerCount == 0u ||
        sourceBytesPerRow == 0u || sourceBytesPerImage == 0u ||
        sourceSize.width == 0u || sourceSize.height == 0u ||
        sourceSize.depth == 0u ||
        (layerCount > 1u && sourceLayerStride == 0u)) {
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

        if (mglRenderCppEncodeTextureUploadLayersForCommandBufferOwner(
                _renderPassManager.state->currentCommandBufferOwner,
                (__bridge void *)sourceBuffer, sourceOffset,
                sourceBytesPerRow, sourceBytesPerImage, sourceLayerStride,
                sourceSize.width, sourceSize.height, sourceSize.depth,
                (__bridge void *)texture, destinationSlice, layerCount,
                destinationLevel, destinationOrigin.x,
                destinationOrigin.y, destinationOrigin.z) != 0) {
            NSLog(@"MGL ERROR: C++ ordered upload encode failed (%s)",
                  reason ? reason : "texture_upload");
            [self recordGPUError];
            return false;
        }

        return true;
    }

    MGLMetalCommandBufferRef uploadCB = mglTextureCreateCommandBuffer(_commandQueue);
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

    if (mglRenderCppEncodeTextureUploadLayers(
            (__bridge void *)uploadCB, (__bridge void *)sourceBuffer,
            sourceOffset, sourceBytesPerRow, sourceBytesPerImage,
            sourceLayerStride,
            sourceSize.width, sourceSize.height, sourceSize.depth,
            (__bridge void *)texture, destinationSlice, layerCount,
            destinationLevel, destinationOrigin.x,
            destinationOrigin.y, destinationOrigin.z) != 0) {
        NSLog(@"MGL ERROR: C++ dedicated upload encode failed (%s)",
              reason ? reason : "texture_upload");
        [self recordGPUError];
        return false;
    }

    dispatch_semaphore_t completionSemaphore = kMGLSynchronizeTextureUploads
        ? dispatch_semaphore_create(0)
        : NULL;
    __block BOOL uploadError = NO;
    __weak typeof(self) weakSelf = self;
    mglRenderAddCommandBufferCompletion(
        uploadCB,
        ^(const MGLRenderCppCommandBufferState *uploadState) {
        if (uploadState->has_error) {
            uploadError = YES;
            NSLog(@"MGL ERROR: dedicated upload command buffer failed (%s): %@",
                  reason ? reason : "texture_upload",
                  mglRenderCommandBufferErrorString(uploadState));
            [weakSelf recordGPUError];
        }

        if (completionSemaphore) {
            dispatch_semaphore_signal(completionSemaphore);
        }
    });

    mglTextureCommitCommandBuffer(uploadCB);

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

    return !uploadError;
}

- (bool)uploadTextureSliceViaBlit:(MGLMetalTextureRef)texture
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
    MGLRenderCppTextureUploadPlan uploadPlan = {0};
    if (mglRenderCppBuildTextureUploadPlan(
            (uint32_t)texTarget, (uint32_t)textureType,
            (uint32_t)texture.storageMode, (uint32_t)texture.pixelFormat,
            MGLCapabilityHasBug(&_capability,
                                MGL_BUG_3D_COPY_FROM_BUFFER_SLICE_OOB) ? 1 : 0,
            width, height, depth, bytesPerRow, bytesPerImage, level, slice,
            &uploadPlan) != 0) {
        NSLog(@"MGL WARNING: Rejecting invalid texture upload plan (tex=%u target=0x%x level=%lu slice=%lu)",
              (unsigned)texName, (unsigned)texTarget, (unsigned long)level,
              (unsigned long)slice);
        return false;
    }

    if (textureType == MTLTextureTypeCube || textureType == MTLTextureTypeCubeArray) {
        static uint64_t s_cubeUploadLogs = 0;
        uint64_t hit = ++s_cubeUploadLogs;
        if (hit <= 4ull || (hit % 2048ull) == 0ull) {
            NSLog(@"MGL CUBE UPLOAD tex=%u glTarget=0x%x face=%lu slice=%lu level=%lu origin=(0,0,0) size=%lux%lux%lu bpr=%lu bpi=%lu ptr=%p",
                  texName,
                  texTarget,
                  (unsigned long)slice,
                  (unsigned long)slice,
                  (unsigned long)level,
                  (unsigned long)width,
                  (unsigned long)uploadPlan.normalized_height,
                  (unsigned long)uploadPlan.copy_depth,
                  (unsigned long)bytesPerRow,
                  (unsigned long)uploadPlan.normalized_bytes_per_image,
                  bytes);
        }
    }

    const uint32_t uploadRoute = uploadPlan.route;

    /* 1D texture upload via replaceRegion branch:
     * - 1D textures are a low-frequency update path; replaceRegion is safe in this scenario;
     * - Before entering this function, the caller has already flushed CPU-side deferred
     *   draws via mglFlushPendingDrawsBeforeTextureWrite, avoiding ordering races between
     *   the upload and uncommitted render command buffers;
     * - Only available for shared storage; Private storage (e.g. MSAA) must fall back to the blit path. */
    if (uploadRoute == MGL_RENDER_CPP_TEXTURE_UPLOAD_ROUTE_REPLACE_1D) {
        @try {
            MTLRegion region = uploadPlan.replace_region_dimension == 1u
                ? MTLRegionMake1D(0, width)
                : MTLRegionMake2D(0, 0, width,
                                  uploadPlan.normalized_height);
            mglTextureReplaceRegion(
                texture, region, uploadPlan.destination_level,
                uploadPlan.destination_slice, bytes, bytesPerRow,
                uploadPlan.normalized_bytes_per_image,
                uploadPlan.replace_use_slice != 0u);
            if (mglTraceLogIsEnabled() &&
                texture.pixelFormat == MTLPixelFormatR8Unorm &&
                width > 0) {
                const uint8_t *first = (const uint8_t *)bytes;
                mglTraceLog("TEXTURE_UPLOAD_1D_REPLACE tex=%u target=0x%x mtlType=%lu size=%lux%lu bpr=%lu bpi=%lu first=%u",
                            (unsigned)texName,
                            (unsigned)texTarget,
                            (unsigned long)textureType,
                            (unsigned long)width,
                            (unsigned long)uploadPlan.normalized_height,
                            (unsigned long)bytesPerRow,
                            (unsigned long)uploadPlan.normalized_bytes_per_image,
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
     * - Metal requires bytesPerImage for 3D replaceRegion uploads, so padded
     *   depth planes are repacked and uploaded with the tight image stride.
     * - Only shared storage supports replaceRegion.  Do not fall back to the
     *   known-bad copyFromBuffer path while the AGX bug marker is active.
     * - The route already rejects private storage while the bug marker is up;
     *   reaching this branch means the upload is shared and must use
     *   replaceRegion (never the known-bad blit). */
    if (uploadRoute == MGL_RENDER_CPP_TEXTURE_UPLOAD_ROUTE_REPLACE_3D) {
        const void *replaceBytes = bytes;
        void *tightlyPackedBytes = NULL;
        if (uploadPlan.requires_repack) {
            /* P4.4: depth-plane repack 在 C++（mglRenderCppTextureRepackDepthPlanes
             * —— 纯数据变换，两门共用，无 A/B 分歧）。溢出/分配失败返回 NULL，
             * 与内联版逐点等价。 */
            tightlyPackedBytes = mglRenderCppTextureRepackDepthPlanes(
                bytes, uploadPlan.normalized_bytes_per_image,
                uploadPlan.expected_bytes_per_image, uploadPlan.copy_depth);
            if (!tightlyPackedBytes) {
                return false;
            }
            replaceBytes = tightlyPackedBytes;
        }

        @try {
            MTLRegion region = MTLRegionMake3D(
                0, 0, 0, width, uploadPlan.normalized_height,
                uploadPlan.copy_depth);
            mglTextureReplaceRegion(
                texture, region, uploadPlan.destination_level,
                uploadPlan.destination_slice, replaceBytes, bytesPerRow,
                uploadPlan.expected_bytes_per_image, YES);
            free(tightlyPackedBytes);
            return true;
        } @catch (NSException *exception) {
            free(tightlyPackedBytes);
            NSLog(@"MGL WARNING: 3D texture replaceRegion upload failed (tex=%u level=%lu): %@",
                  (unsigned)texName, (unsigned long)level,
                  exception.reason);
            return false;
        }
    }

    /* 3D + Private while the AGX copyFromBuffer workaround is required:
     * rejected by the C++ route (blit is known-bad and replaceRegion does
     * not support private storage). */
    if (uploadRoute == MGL_RENDER_CPP_TEXTURE_UPLOAD_ROUTE_REJECT) {
        NSLog(@"MGL WARNING: Rejecting private 3D upload while AGX copyFromBuffer workaround is required (tex=%u level=%lu)",
              (unsigned)texName, (unsigned long)level);
        return false;
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
    void *stagingOwner = NULL;
    void *borrowedStagingBuffer = NULL;
    __unsafe_unretained MGLMetalBufferRef uploadBuffer = nil;
    if (mglRenderCppCreateTextureStagingOwner(
            bytes, uploadPlan.buffer_size, MTLResourceStorageModeShared,
            &stagingOwner, &borrowedStagingBuffer) == 0 &&
        stagingOwner && borrowedStagingBuffer) {
        uploadBuffer = (__bridge MGLMetalBufferRef)borrowedStagingBuffer;
    }
    if (!uploadBuffer) {
        mglRenderCppDestroyTextureStagingOwner(&stagingOwner);
        NSLog(@"MGL WARNING: Failed to allocate upload buffer for texture blit");
        return false;
    }

    bool uploaded = [self copyTextureUploadWithDedicatedCommandBuffer:uploadBuffer
                                                         sourceOffset:0
                                                    sourceBytesPerRow:bytesPerRow
                                                  sourceBytesPerImage:uploadPlan.normalized_bytes_per_image
                                                   sourceLayerStride:0
                                                           layerCount:1
                                                            sourceSize:MTLSizeMake(width, uploadPlan.normalized_height, uploadPlan.copy_depth)
                                                             toTexture:texture
                                                      destinationSlice:uploadPlan.destination_slice
                                                      destinationLevel:uploadPlan.destination_level
                                                     destinationOrigin:MTLOriginMake(0, 0, 0)
                                                                reason:"texture_upload_blit"];
    /* The encoded blit retains its source resource until command-buffer
     * completion; release the C++ staging owner as soon as encoding ends. */
    mglRenderCppDestroyTextureStagingOwner(&stagingOwner);
    if (!uploaded) {
        NSLog(@"MGL WARNING: Dedicated texture upload failed (level=%lu slice=%lu)",
              (unsigned long)level, (unsigned long)slice);
    }
    return uploaded;
}

- (bool)uploadFullCPUTextureDataIntoTexture:(Texture *)tex
                                      metal:(MGLMetalTextureRef)texture
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

    /* P4.5 (item 1111/1116): 逐 level 迭代 + 可上传判定 + 数据准备 + 短后备
     * 分类在 C++（mglRenderCppBuildLevelUploadOps——纯数据变换，两门共用）；
     * ObjC 只负责上传每个 op + 诊断日志 + 释放自有数据。单面（2D）专用。 */
    MGLRenderCppLevelUploadOp uploadOps[levelCount ? levelCount : 1u];
    uint32_t opCount = 0;
    uint32_t shortCount = 0;
    uint32_t badCount = 0;
    if (mglRenderCppBuildLevelUploadOps(
            tex->faces[0].levels, levelCount,
            (uint32_t)texture.textureType,
            (uint32_t)tex->internalformat,
            (uint32_t)texture.pixelFormat,
            uploadOps, levelCount,
            &opCount, &shortCount, &badCount) != 0) {
        return false;
    }

    bool uploadedAny = false;
    bool failedAny = (shortCount + badCount) > 0;
    for (uint32_t i = 0; i < opCount; i++) {
        MGLRenderCppLevelUploadOp *op = &uploadOps[i];
        if (op->kind == 1u) {
            static uint64_t s_shortBackingLogs = 0;
            uint64_t hit = ++s_shortBackingLogs;
            if (kMGLDiagnosticStateLogs &&
                (hit <= 32ull || (hit % 512ull) == 0ull)) {
                mglTraceLogNSString(@"MGL TEXTURE CPU-REFRESH skip short backing tex=%u level=%u face=0 have=%llu need=%llu reason=%s hit=%llu",
                              (unsigned)tex->name,
                              (unsigned)op->level,
                              (unsigned long long)op->available_bytes,
                              (unsigned long long)op->needed_bytes,
                              reason ? reason : "(null)",
                              (unsigned long long)hit);
            }
            continue;
        }

        bool uploaded = [self uploadTextureSliceViaBlit:texture
                                                texName:tex->name
                                             texTarget:tex->target
                                                 bytes:op->data
                                           bytesPerRow:(NSUInteger)op->bytes_per_row
                                         bytesPerImage:(NSUInteger)op->bytes_per_image
                                                 width:(NSUInteger)op->width
                                                height:(NSUInteger)op->height
                                                 depth:(NSUInteger)op->copy_depth
                                                 level:op->level
                                                 slice:0];
        if (op->owns_data) {
            free((void *)op->data);
        }
        if (uploaded) {
            uploadedAny = true;
        } else {
            failedAny = true;
        }
    }

    static uint64_t s_refreshLogs = 0;
    uint64_t hit = ++s_refreshLogs;
    if (kMGLDiagnosticStateLogs &&
        (uploadedAny || hit <= 32ull || (hit % 512ull) == 0ull)) {
        mglTraceLogNSString(@"MGL TEXTURE CPU-REFRESH tex=%u mtl=%p uploaded=%d failed=%d dirty=0x%x levels=%u reason=%s hit=%llu",
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

- (void)mglApplyPendingDefaultColorClearToTexture:(MGLMetalTextureRef)texture
{
    if (!ctx || !texture || !(ctx->state.default_fbo_clear_bitmask & GL_COLOR_BUFFER_BIT)) {
        return;
    }

    if (mglRenderCppEncodeColorClearForCommandBufferOwner(
            _renderPassManager.state->currentCommandBufferOwner,
            (__bridge void *)texture, 0, 0, 0,
            ctx->state.default_clear_color[0],
            ctx->state.default_clear_color[1],
            ctx->state.default_clear_color[2],
            ctx->state.default_clear_color[3]) == 0) {
        ctx->state.default_fbo_clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
        return;
    }
    NSLog(@"MGL WARNING: C++ default framebuffer color clear failed");
}

- (void)mglApplyPendingFBOColorClearForReadback:(Framebuffer *)fbo
                                     attachment:(FBOAttachment *)attachment
                                    textureObj:(Texture *)textureObj
                                     mtlTexture:(MGLMetalTextureRef)texture
                                  attachmentEnum:(GLenum)attachmentEnum
{
    (void)attachmentEnum;
    if (!fbo || !attachment || !texture || !(attachment->clear_bitmask & GL_COLOR_BUFFER_BIT)) {
        return;
    }

    MGLMetalAttachmentSubresource subresource =
        mglMetalAttachmentSubresourceForAttachment(attachment);
    if (mglRenderCppEncodeColorClearForCommandBufferOwner(
            _renderPassManager.state->currentCommandBufferOwner,
            (__bridge void *)texture, subresource.level,
            subresource.slice, subresource.depthPlane,
            attachment->clear_color[0], attachment->clear_color[1],
            attachment->clear_color[2], attachment->clear_color[3]) == 0) {
        attachment->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
        mglMarkTextureLevelRenderTargetWritten(
            textureObj, attachment->level);
        return;
    }
    NSLog(@"MGL WARNING: C++ readPixels FBO color clear failed fbo=%u",
          (unsigned)fbo->name);
}

/* P4.5 (item 1171): readback 共享序列——staging buffer 创建 + blit encoder +
 * copy + end（带异常清理）+ completion-wait（0.25s 超时 + error 状态）+ commit
 * + newCommandBuffer。颜色/深度两条 readPixels 路径的编排逐点一致，仅日志
 * 主语（logKind）与转换不同。返回共享 staging buffer（等待后 contents 可用）；
 * 失败返回 nil（已按调用点语义置 GL 错误）。outSuccess 为 NO 表示等待超时/
 * CB 错误（buffer 仍有效但内容不可信，调用方跳过转换）。 */
- (MGLMetalBufferRef)readbackStageAndWaitTexture:(MGLMetalTextureRef)sourceTexture
                                 sourceLevel:(NSUInteger)sourceLevel
                                 sourceSlice:(NSUInteger)sourceSlice
                             sourceDepthPlane:(NSUInteger)sourceDepthPlane
                                   copyOrigin:(MTLOrigin)copyOrigin
                                     copySize:(MTLSize)copySize
                         stagingBytesPerRow:(NSUInteger)stagingBytesPerRow
                                stagingSize:(NSUInteger)stagingSize
                                     reason:(const char *)reason
                                    logKind:(const char *)logKind
                                     success:(BOOL *)outSuccess
{
    if (outSuccess) {
        *outSuccess = YES;
    }

    MGLMetalBufferRef readBuffer = mglTextureCreateBuffer(
        _device, stagingSize, MTLResourceStorageModeShared);
    MGLMetalBlitCommandEncoderRef blitEncoder = readBuffer
        ? mglRenderCreateBlitEncoderForCommandBufferOwner(
              _renderPassManager.state->currentCommandBufferOwner)
        : nil;
    if (!readBuffer || !blitEncoder) {
        NSLog(@"MGL WARNING: readPixels failed to create %s resources for %s",
              logKind ? logKind : "readback",
              reason ? reason : "unknown");
        mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return nil;
    }

    BOOL blitEncoderEnded = NO;
    @try {
        mglTextureCopyTextureToBuffer(
            blitEncoder, sourceTexture, sourceSlice, sourceLevel,
            copyOrigin, copySize,
            readBuffer, 0u, stagingBytesPerRow, stagingSize);
        mglTextureEndBlitEncoder(blitEncoder);
        blitEncoderEnded = YES;
    } @catch (NSException *exception) {
        if (!blitEncoderEnded) {
            @try {
                mglTextureEndBlitEncoder(blitEncoder);
            } @catch (NSException *endException) {
                NSLog(@"MGL WARNING: readPixels failed to end %s blit encoder after copy exception for %s: %@",
                      logKind ? logKind : "readback",
                      reason ? reason : "unknown",
                      endException);
            }
        }
        NSLog(@"MGL WARNING: readPixels %s texture copy failed for %s: %@",
              logKind ? logKind : "readback",
              reason ? reason : "unknown",
              exception);
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return nil;
    }

    MGLMetalCommandBufferRef readbackCommandBuffer =
        [_renderPassManager detachCurrentCommandBufferForSubmission];
    MGLRenderCppCommandBufferTransaction readbackTransaction = {0};
    int readbackTransactionResult = [_renderPassManager
        commitCommandBufferTransaction:readbackCommandBuffer
        recoveryOwner:_gpuRecovery.commandRecoveryOwner
        waitForCompletion:YES
        result:&readbackTransaction];
    if (readbackTransactionResult != 0 || readbackTransaction.has_error) {
        NSLog(@"MGL WARNING: readPixels %s owner transaction failed for %s",
              logKind ? logKind : "readback", reason ? reason : "unknown");
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        if (outSuccess) {
            *outSuccess = NO;
        }
    }

    if (readbackTransactionResult == 0 &&
        readbackTransaction.completion.has_error) {
        NSLog(@"MGL WARNING: readPixels %s command buffer failed for %s: %@; returning zeroed data",
              logKind ? logKind : "readback",
              reason ? reason : "unknown",
              mglRenderCommandBufferErrorString(
                  &readbackTransaction.completion));
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        if (outSuccess) {
            *outSuccess = NO;
        }
    }

    [_renderPassManager releaseDetachedCommandBufferIfOwned:readbackCommandBuffer];
    [self newCommandBuffer];
    return readBuffer;
}

- (BOOL)mglReadColorTextureAsBGRA8:(MGLMetalTextureRef)sourceTexture
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

    /* P4.5 (item 1141/887): readPixels 区域对 level 尺寸的裁剪（含空
     * 判断与 Metal 源 Y 翻转）在 C++
     * （mglRenderCppReadTextureRegionClip，两门共用）。 */
    MGLRenderCppReadTextureRegionClip clip = {0};
    mglRenderCppReadTextureRegionClip(
        (int64_t)region.origin.x, (int64_t)region.origin.y,
        (int64_t)region.size.width, (int64_t)region.size.height,
        (int64_t)levelWidth, (int64_t)levelHeight, &clip);
    NSInteger copyW = (NSInteger)clip.copy_w;
    NSInteger copyH = (NSInteger)clip.copy_h;
    NSInteger dstX = (NSInteger)clip.dst_x;
    NSInteger dstY = (NSInteger)clip.dst_y;
    NSInteger metalSrcX = (NSInteger)clip.metal_src_x;
    NSInteger metalSrcY = (NSInteger)clip.metal_src_y;
    if (clip.empty) {
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

    BOOL readbackSuccess = YES;
    MGLMetalBufferRef readBuffer = [self readbackStageAndWaitTexture:sourceTexture
                                                     sourceLevel:sourceLevel
                                                     sourceSlice:sourceSlice
                                                 sourceDepthPlane:sourceDepthPlane
                                                       copyOrigin:MTLOriginMake((NSUInteger)metalSrcX,
                                                                                (NSUInteger)metalSrcY,
                                                                                sourceDepthPlane)
                                                         copySize:MTLSizeMake((NSUInteger)copyW,
                                                                              (NSUInteger)copyH,
                                                                              1u)
                                             stagingBytesPerRow:stagingBytesPerRow
                                                    stagingSize:stagingSize
                                                         reason:reason
                                                        logKind:"readback"
                                                         success:&readbackSuccess];
    if (!readBuffer) {
        return NO;
    }

    if (readbackSuccess) {
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

    return readbackSuccess;
}

- (BOOL)mglReadDepthTextureAsFloat:(MGLMetalTextureRef)sourceTexture
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

    /* P4.5 (item 1141/887): readPixels 区域对 level 尺寸的裁剪（含空
     * 判断与 Metal 源 Y 翻转）在 C++
     * （mglRenderCppReadTextureRegionClip，两门共用）。 */
    MGLRenderCppReadTextureRegionClip clip = {0};
    mglRenderCppReadTextureRegionClip(
        (int64_t)region.origin.x, (int64_t)region.origin.y,
        (int64_t)region.size.width, (int64_t)region.size.height,
        (int64_t)levelWidth, (int64_t)levelHeight, &clip);
    NSInteger copyW = (NSInteger)clip.copy_w;
    NSInteger copyH = (NSInteger)clip.copy_h;
    NSInteger dstX = (NSInteger)clip.dst_x;
    NSInteger dstY = (NSInteger)clip.dst_y;
    NSInteger metalSrcX = (NSInteger)clip.metal_src_x;
    NSInteger metalSrcY = (NSInteger)clip.metal_src_y;
    if (clip.empty) {
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

    BOOL readbackSuccess = YES;
    MGLMetalBufferRef readBuffer = [self readbackStageAndWaitTexture:sourceTexture
                                                     sourceLevel:sourceLevel
                                                     sourceSlice:sourceSlice
                                                 sourceDepthPlane:sourceDepthPlane
                                                       copyOrigin:MTLOriginMake((NSUInteger)metalSrcX,
                                                                                (NSUInteger)metalSrcY,
                                                                                sourceDepthPlane)
                                                         copySize:MTLSizeMake((NSUInteger)copyW,
                                                                              (NSUInteger)copyH,
                                                                              1u)
                                             stagingBytesPerRow:stagingBytesPerRow
                                                    stagingSize:stagingSize
                                                         reason:reason
                                                        logKind:"depth readback"
                                                         success:&readbackSuccess];
    if (!readBuffer) {
        return NO;
    }

    if (readbackSuccess) {
        uint8_t *dst = ((uint8_t *)pixelBytes) + dstOffset;
        if (sourceIsDepthStencil || sourceIsDepth16) {
            /* P4.5 (item 1171): Depth16 / unpacked depth-float -> GL float. */
            mglRenderCppCopyDepthTextureBytesToFloat(
                readBuffer.contents, (uint64_t)stagingBytesPerRow,
                dst, (uint64_t)bytesPerRow,
                (uint64_t)copyW, (uint64_t)copyH,
                (uint64_t)sourceDepthBytes,
                sourceIsDepth16 ? 1 : 0,
                1);
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

    return readbackSuccess;
}

- (BOOL)mglReadIntegerTextureAsRGBA32:(MGLMetalTextureRef)sourceTexture
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
    /* P4.5 (item 1171/1116): 源格式分类（19 项 MTLPixelFormat ->
     * {分量数, 分量字节, 有符号, RGB10A2} 表）在 C++
     * （mglRenderCppIntegerReadbackSourceClassify，纯分类，两门共用）。 */
    MGLRenderCppIntegerReadbackSource src = {0};
    mglRenderCppIntegerReadbackSourceClassify(
        (uint32_t)sourceTexture.pixelFormat, &src);
    if (!src.recognized) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }
    NSUInteger componentCount = (NSUInteger)src.component_count;
    NSUInteger sourceComponentBytes = (NSUInteger)src.component_bytes;
    BOOL sourceSigned = src.source_signed != 0;
    BOOL sourceRGB10A2Uint = src.source_rgb10a2_uint != 0;

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

    /* P4.5 (item 1171/1116): packed 类型表（10 项 GL packed type ->
     * {位宽, 移位, 输出字节, 输出分量}）在 C++
     * （mglRenderCppIntegerReadbackPackedTypeClassify，纯分类，两门共用）。
     * packedTotalBits 已无读取方（round-54 转换迁移后为死变量），不迁移。 */
    MGLRenderCppIntegerPackedType packed = {0};
    mglRenderCppIntegerReadbackPackedTypeClassify((uint32_t)packedType, &packed);
    BOOL isPackedType = packed.is_packed != 0;
    uint32_t packedBitWidths[4] = {
        packed.bit_widths[0], packed.bit_widths[1],
        packed.bit_widths[2], packed.bit_widths[3]};
    uint32_t packedShifts[4] = {
        packed.shifts[0], packed.shifts[1],
        packed.shifts[2], packed.shifts[3]};
    NSUInteger packedOutputBytes = (NSUInteger)packed.output_bytes;
    if (packed.output_components > 0u) {
        outputComponents = (NSUInteger)packed.output_components;
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

    MGLMetalBufferRef readBuffer = mglTextureCreateBuffer(
        _device, stagingSize, MTLResourceStorageModeShared);
    MGLMetalBlitCommandEncoderRef blit = readBuffer
        ? mglRenderCreateBlitEncoderForCommandBufferOwner(
              _renderPassManager.state->currentCommandBufferOwner)
        : nil;
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
    mglTextureCopyTextureToBuffer(
        blit, sourceTexture, mtlSlice, mipmapLevel,
        MTLOriginMake((NSUInteger)minX, blitSrcY, 0u),
        MTLSizeMake((NSUInteger)copyW, (NSUInteger)copyH, 1u), readBuffer,
        0u, srcBytesPerRow, stagingSize);
    mglTextureEndBlitEncoder(blit);
    MGLMetalCommandBufferRef integerReadbackCommandBuffer =
        [_renderPassManager detachCurrentCommandBufferForSubmission];
    MGLRenderCppCommandBufferTransaction integerReadbackTransaction = {0};
    int integerReadbackResult = [_renderPassManager
        commitCommandBufferTransaction:integerReadbackCommandBuffer
        recoveryOwner:_gpuRecovery.commandRecoveryOwner
        waitForCompletion:YES
        result:&integerReadbackTransaction];
    if (integerReadbackResult != 0 || integerReadbackTransaction.has_error) {
        NSLog(@"MGL ERROR: integer texture readback owner transaction failed");
        [_renderPassManager releaseDetachedCommandBufferIfOwned:integerReadbackCommandBuffer];
        return NO;
    }
    [_renderPassManager releaseDetachedCommandBufferIfOwned:integerReadbackCommandBuffer];

    NSUInteger dstX = (NSUInteger)(minX - (NSInteger)region.origin.x);
    NSUInteger dstY = (NSUInteger)(minY - (NSInteger)region.origin.y);
    /* P4.5 (item 1171/1116): 像素级转换（分量提取 + GL_INTEGER 打包/钳制 +
     * 行拷贝）在 C++（mglRenderCppConvertIntegerReadback——纯数据变换，两门
     * 共用，语义与内联版逐点等价）。src/dst 参数与上面的 staging/等待逻辑
     * 一致；blit 源 Y-flip 逻辑保持不变（isRenderTarget 条件）。 */
    MGLRenderCppIntegerReadbackConvertParams convert = {
        .src = (const uint8_t *)readBuffer.contents,
        .src_bytes_per_row = srcBytesPerRow,
        .source_component_count = (uint32_t)componentCount,
        .source_component_bytes = (uint32_t)sourceComponentBytes,
        .source_signed = sourceSigned ? 1 : 0,
        .source_rgb10a2_uint = sourceRGB10A2Uint ? 1 : 0,
        .copy_w = (uint32_t)copyW,
        .copy_h = (uint32_t)copyH,
        .dst = (uint8_t *)pixelBytes,
        .dst_bytes_per_row = bytesPerRow,
        .dst_pixel_bytes = dstPixelBytes,
        .dst_x = (uint64_t)dstX,
        .dst_y = (uint64_t)dstY,
        .output_components = (uint32_t)outputComponents,
        .component_map = componentMap,
        .output_component_bytes = (uint32_t)outputComponentBytes,
        .packed_type = (uint32_t)packedType,
        .is_packed_type = isPackedType ? 1 : 0,
        .packed_bit_widths = packedBitWidths,
        .packed_shifts = packedShifts,
        .packed_output_bytes = (uint32_t)packedOutputBytes,
    };
    if (mglRenderCppConvertIntegerReadback(&convert) != 0) {
        [self newCommandBuffer];
        return NO;
    }

    [self newCommandBuffer];
    return YES;
}

- (void)mglApplyPendingFBODepthClearForReadback:(Framebuffer *)fbo
                                     attachment:(FBOAttachment *)attachment
                                     textureObj:(Texture *)textureObj
                                     mtlTexture:(MGLMetalTextureRef)texture
{
    if (!fbo || !attachment || !texture || !(attachment->clear_bitmask & GL_DEPTH_BUFFER_BIT)) {
        return;
    }

    MGLMetalAttachmentSubresource subresource =
        mglMetalAttachmentSubresourceForAttachment(attachment);
    if (mglRenderCppEncodeDepthClearForCommandBufferOwner(
            _renderPassManager.state->currentCommandBufferOwner,
            (__bridge void *)texture, subresource.level,
            subresource.slice, subresource.depthPlane,
            attachment->clear_color[0]) == 0) {
        attachment->clear_bitmask &= ~GL_DEPTH_BUFFER_BIT;
        mglMarkTextureLevelRenderTargetWritten(textureObj, attachment->level);
    } else {
        NSLog(@"MGL WARNING: C++ readPixels depth clear failed fbo=%u",
              (unsigned)fbo->name);
    }
}

- (void)mglApplyPendingDefaultDepthClearToTexture:(MGLMetalTextureRef)texture
{
    if (!ctx || !texture || !(ctx->state.default_fbo_clear_bitmask & GL_DEPTH_BUFFER_BIT)) {
        return;
    }

    if (mglRenderCppEncodeDepthClearForCommandBufferOwner(
            _renderPassManager.state->currentCommandBufferOwner,
            (__bridge void *)texture, 0, 0, 0,
            ctx->state.var.depth_clear_value) == 0) {
        ctx->state.default_fbo_clear_bitmask &= ~GL_DEPTH_BUFFER_BIT;
    } else {
        NSLog(@"MGL WARNING: C++ default depth clear failed");
    }
}

- (void)mtlReadDepthPixels:(GLMContext)glm_ctx
                pixelBytes:(void *)pixelBytes
               bytesPerRow:(NSUInteger)bytesPerRow
             bytesPerImage:(NSUInteger)bytesPerImage
                fromRegion:(MTLRegion)region
{
    MGL_ASSERT_GL_THREAD();
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

        MGLMetalTextureRef texture = (__bridge MGLMetalTextureRef)(readTextureObject->mtl_data);
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
    MGLMetalTextureRef texture = nil;
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

    MGLMetalTextureRef texture = (__bridge MGLMetalTextureRef)(textureObj->mtl_data);
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

        MGLMetalTextureRef texture = (__bridge MGLMetalTextureRef)(readTextureObject->mtl_data);
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
    MGLMetalTextureRef texture = nil;

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
            (void)[self mglApplyPendingDrawableSize];
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
    MGLMetalTextureRef texture = nil;

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

    texture = (__bridge MGLMetalTextureRef)(tex->mtl_data);
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
    if (mglRenderCppCommandBufferOwnerHasCurrent(
            _renderPassManager.state->currentCommandBufferOwner) == 1) {
        MGLMetalCommandBufferRef pendingCB =
            [_renderPassManager detachCurrentCommandBufferForSubmission];
        @try {
            [self commitCommandBufferWithAGXRecovery:pendingCB];
            mglTextureWaitCommandBuffer(pendingCB);
        } @catch (NSException *e) {
            NSLog(@"MGL WARNING: mtlGetTexImage pre-readback flush failed: %@", e.reason);
        }
        MGLRenderCppCommandBufferState pendingState =
            mglRenderCommandBufferState(pendingCB);
        if (pendingState.has_error) {
            NSLog(@"MGL WARNING: mtlGetTexImage pre-readback command buffer error: %@",
                  mglRenderCommandBufferErrorString(&pendingState));
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
    /* P4.5 (item 1171/1116): integer-readback 分类（源整数格式表 +
     * GL_*_INTEGER 输出判定 + 分量映射（BGR/BGRA 序 + GREEN/BLUE/ALPHA
     * 单分量兼容枚举）+ 按类型的分量字节数）在 C++
     * （mglRenderCppIntegerReadbackClassify，纯分类，两门共用）。 */
    MGLRenderCppIntegerReadbackClassify classify = {0};
    mglRenderCppIntegerReadbackClassify(
        (uint32_t)texture.pixelFormat, (uint32_t)format, (uint32_t)type,
        &classify);

    if (classify.source_is_integer_texture &&
        classify.output_is_integer_format) {
        /* Pass the original (non-Y-flipped) region. mglReadIntegerTextureAsRGBA32
         * does its own Y-flip on the blit source origin AND Y-flips the output
         * rows, so passing a pre-Y-flipped readRegion here would double-flip. */
        [self mglReadIntegerTextureAsRGBA32:texture
                                pixelBytes:pixelBytes
                               bytesPerRow:bytesPerRow
                             bytesPerImage:bytesPerImage
                                fromRegion:region
                          outputComponents:(NSUInteger)classify.output_components
                       outputComponentBytes:(NSUInteger)classify.output_component_bytes
                              componentMap:classify.component_map
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
        /* P4.5 (item 1171/1116): 直接 R32F 判定 + BGRA8 转换资格 + 源
         * BGRA8 族判定 + row/image/total 字节计算（含非 BGRA8 源按源 bpp
         * 计 pitch、depth>1 + bytesPerImage 情形）在 C++
         * （mglRenderCppGetTexImagePlan，两门共用）。 */
        MGLRenderCppGetTexImagePlan plan = {0};
        mglRenderCppGetTexImagePlan(
            (uint32_t)texture.pixelFormat,
            (uint32_t)format,
            (uint32_t)type,
            (uint32_t)readRegion.size.width,
            (uint32_t)readRegion.size.height,
            (uint32_t)readRegion.size.depth,
            (uint32_t)dstPixelBytes,
            (uint32_t)mglMetalReadbackBytesPerPixel(texture.pixelFormat),
            mglMetalReadbackFormatIsBGRA8Compatible(texture.pixelFormat) ? 1 : 0,
            (uint32_t)bytesPerRow,
            (uint32_t)bytesPerImage,
            1,
            &plan);
        useBGRA8Conversion = plan.use_bgra8_conversion;
        NSUInteger rowBytes = (NSUInteger)plan.row_bytes;
        NSUInteger imageBytes = (NSUInteger)plan.image_bytes;
        NSUInteger totalBytes = (NSUInteger)plan.total_bytes;

        MGLMetalBufferRef stagingBuffer = mglTextureCreateBuffer(
            _device, totalBytes, MTLResourceStorageModeShared);
        if (!stagingBuffer) {
            NSLog(@"MGL ERROR: mtlGetTexImage failed to allocate staging buffer for texture %u", tex->name);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
            return;
        }

        MGLMetalCommandBufferRef blitCB = mglTextureCreateCommandBuffer(_commandQueue);
        if (!blitCB) {
            NSLog(@"MGL ERROR: mtlGetTexImage failed to create blit command buffer for texture %u", tex->name);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        MGLMetalBlitCommandEncoderRef blitEncoder = mglTextureCreateBlitEncoder(blitCB);
        if (!blitEncoder) {
            NSLog(@"MGL ERROR: mtlGetTexImage failed to create blit encoder for texture %u", tex->name);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        mglTextureCopyTextureToBuffer(
            blitEncoder, texture, slice, level, readRegion.origin,
            readRegion.size, stagingBuffer, 0, rowBytes, imageBytes);

        mglTextureEndBlitEncoder(blitEncoder);
        mglTextureCommitCommandBuffer(blitCB);
        mglTextureWaitCommandBuffer(blitCB);

        MGLRenderCppCommandBufferState blitState =
            mglRenderCommandBufferState(blitCB);
        if (blitState.has_error) {
            NSLog(@"MGL ERROR: mtlGetTexImage blit failed for texture %u: %@",
                  tex->name, mglRenderCommandBufferErrorString(&blitState));
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
	            /* P4.5 (item 1171/1116): row pitch（转换路径：非 BGRA8 源按
	             * 源 bpp，BGRA8 源 4B；否则 bytesPerRow 或 width*max(dst,1)）
	             * 在 C++（mglRenderCppGetTexImagePlan，与 private 路径共用）。 */
	            MGLRenderCppGetTexImagePlan plan = {0};
	            mglRenderCppGetTexImagePlan(
	                (uint32_t)texture.pixelFormat,
	                (uint32_t)format,
	                (uint32_t)type,
	                (uint32_t)readRegion.size.width,
	                (uint32_t)readRegion.size.height,
	                (uint32_t)readRegion.size.depth,
	                (uint32_t)dstPixelBytes,
	                (uint32_t)mglMetalReadbackBytesPerPixel(texture.pixelFormat),
	                mglMetalReadbackFormatIsBGRA8Compatible(texture.pixelFormat) ? 1 : 0,
	                (uint32_t)bytesPerRow,
	                (uint32_t)bytesPerImage,
	                0,
	                &plan);
	            NSUInteger rowBytes = (NSUInteger)plan.row_bytes;
	            NSUInteger totalBytes = (NSUInteger)plan.image_bytes;
            NSMutableData *readback = [NSMutableData dataWithLength:totalBytes];
            if (!readback) {
                mglDispatchError(glm_ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
                return;
            }
            mglTextureGetBytes(
                texture, readback.mutableBytes, rowBytes, bytesPerImage,
                readRegion, level, slice, YES);
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
            mglTextureGetBytes(
                texture, pixelBytes, bytesPerRow, bytesPerImage,
                readRegion, level, slice, YES);
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

    MGLMetalTextureRef texture;

    texture = (__bridge MGLMetalTextureRef)(tex->mtl_data);
    if (!texture) {
        NSLog(@"MGL ERROR: mtlGenerateMipmaps texture %u has no Metal texture after bind", tex->name);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    if (texture.mipmapLevelCount <= 1u) {
        return;
    }

    // start blit encoder
    MGLMetalBlitCommandEncoderRef blitCommandEncoder;
    blitCommandEncoder = mglTextureCreateCurrentBlitEncoder(
        _renderPassManager.state->currentCommandBufferOwner);
    if (!blitCommandEncoder) {
        NSLog(@"MGL ERROR: Failed to create blit encoder for mipmap generation");
        return;
    }

    @try {
        if (mglRenderCppBlitGenerateMipmaps(
                (__bridge void *)blitCommandEncoder,
                (__bridge void *)texture) != 0) {
            [NSException raise:@"MGLGenerateMipmapsError"
                        format:@"C++ mipmap generation failed for texture %u",
                               tex->name];
        }
        mglTextureEndBlitEncoder(blitCommandEncoder);
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: generateMipmapsForTexture failed for texture %u: %@",
              tex->name,
              exception);
        @try {
            mglTextureEndBlitEncoder(blitCommandEncoder);
        } @catch (NSException *endException) {
            NSLog(@"MGL WARNING: failed to end mipmap blit encoder after exception: %@", endException);
        }
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
    }
}

- (bool)encodeTextureBytesUpload:(Texture *)tex
                          source:(MGLMetalBufferRef)buffer
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
    MGL_ASSERT_GL_THREAD();
    if (!tex || !buffer || sourceBytesPerRow == 0 || width == 0 || height == 0) {
        return false;
    }

    if (tex->mtl_data == NULL) {
        [self bindMTLTexture:tex];
        if (tex->mtl_data == NULL) {
            return false;
        }
    }

    MGLMetalTextureRef texture = (__bridge MGLMetalTextureRef)(tex->mtl_data);
    if (!texture) {
        return false;
    }

    MTLTextureType textureType = texture.textureType;
    MGLRenderCppTextureSubUploadPlan uploadPlan = {0};
    if (mglRenderCppTextureSubUploadPlan(
            (uint32_t)tex->target, (uint32_t)textureType, (uint64_t)slice,
            (uint64_t)xoffset, (uint64_t)yoffset, (uint64_t)zoffset,
            (uint64_t)width, (uint64_t)height, (uint64_t)depth,
            (uint64_t)sourceBytesPerRow, (uint64_t)sourceBytesPerImage,
            &uploadPlan) != 0) {
        return false;
    }
    NSUInteger destinationSlice = (NSUInteger)uploadPlan.destination_base_slice;
    MTLOrigin destinationOrigin = MTLOriginMake(
        (NSUInteger)uploadPlan.destination_x,
        (NSUInteger)uploadPlan.destination_y,
        (NSUInteger)uploadPlan.destination_z);
    NSUInteger copyHeight = (NSUInteger)uploadPlan.copy_height;
    NSUInteger copyDepth = (NSUInteger)uploadPlan.copy_depth;
    NSUInteger layerCount = (NSUInteger)uploadPlan.layer_count;
    NSUInteger sourceLayerStride =
        (NSUInteger)uploadPlan.source_layer_stride;
    if (copyHeight > NSUIntegerMax / sourceBytesPerRow) {
        return false;
    }
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
        layerCount > maxDestinationSlices - destinationSlice ||
        destinationOrigin.x > texture.width ||
        destinationOrigin.y > texture.height ||
        destinationOrigin.z > texture.depth ||
        width > texture.width - destinationOrigin.x ||
        copyHeight > texture.height - destinationOrigin.y ||
        copyDepth > texture.depth - destinationOrigin.z) {
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
                                          sourceLayerStride:sourceLayerStride
                                                  layerCount:layerCount
                                                   sourceSize:MTLSizeMake(
                                                       (NSUInteger)uploadPlan.copy_width,
                                                       copyHeight, copyDepth)
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

    MGLMetalBufferRef buffer = (__bridge MGLMetalBufferRef)(buf->data.mtl_data);
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
        MGLMetalTextureRef dstTexture = (__bridge MGLMetalTextureRef)(tex->mtl_data);
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
                            MGLMetalBufferRef uploadBuffer =
                                mglTextureCreateBufferWithBytes(
                                    _device, packedUpload.bytes, packedBytes,
                                    MTLResourceStorageModeShared);
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
    MGLMetalTextureRef dstTexture = (__bridge MGLMetalTextureRef)(tex->mtl_data);
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

    MGLMetalBufferRef uploadBuffer = mglTextureCreateBufferWithBytes(
        _device, packedUpload.bytes, packedBytes,
        MTLResourceStorageModeShared);
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
                                metal:(MGLMetalTextureRef)texture
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

- (void)fillTextureWithSafeInitialContents:(MGLMetalTextureRef)texture
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

                                MGLMetalBufferRef tempBuffer =
                                    mglTextureCreateBufferWithBytes(
                                        _device, properData, fillSize,
                                        MTLResourceStorageModeShared);


                                if (tempBuffer) {

                                    NSLog(@"MGL INFO: Created temporary MTLBuffer for texture data");


                                    if ([self shouldSkipGPUOperations]) {

                                        NSLog(@"MGL AGX: Skipping texture fill during recovery - texture will be empty");

                                    } else {

                                        BOOL uploaded = [self copyTextureUploadWithDedicatedCommandBuffer:tempBuffer

                                                                                              sourceOffset:0

                                                                                         sourceBytesPerRow:properBytesPerRow

                                                                                       sourceBytesPerImage:fillSize

                                                                                        sourceLayerStride:0

                                                                                                layerCount:1

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
                            metal:(MGLMetalTextureRef)texture
                      pixelFormat:(MTLPixelFormat)pixelFormat
                        numFaces:(uint)num_faces
                uploadLevelCount:(GLuint)upload_level_count
                         isArray:(BOOL)is_array
              texture1DBackedBy2D:(BOOL)texture1DBackedBy2D
        texture1DArrayBackedBy2DArray:(BOOL)texture1DArrayBackedBy2DArray
                         texType:(MTLTextureType)tex_type
            outAllLevelsUploaded:(BOOL *)outAllLevelsUploaded
{
    MGL_ASSERT_GL_THREAD();

    if (kMGLDiagnosticStateLogs) {
        mglTraceLogNSString(@"MGL DEBUG: DIRTY_TEXTURE_DATA detected - attempting texture filling");
        mglTraceLogNSString(@"MGL DEBUG: Texture details: target=0x%x, internalformat=0x%x, levels=%d effectiveLevels=%u",
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
                if (hit <= 8ull || (hit % 2048ull) == 0ull) {
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
                                          metal:(MGLMetalTextureRef)texture
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

- (void)fillSmallRGBA8TextureWithGradient:(MGLMetalTextureRef)texture tex:(Texture *)tex
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

                                        mglTextureReplaceRegion(
                                            texture, simpleRegion, 0, 0,
                                            simpleData,
                                            texture.width * sizeof(uint32_t),
                                            texture.width * texture.height * sizeof(uint32_t),
                                            YES);


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
                                    metal:(MGLMetalTextureRef)texture
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
                                       metal:(MGLMetalTextureRef)texture
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


- (void)swizzleTexDesc:(MTLTextureDescriptor *)tex_desc forTex:(Texture*)tex
{
    MTLTextureSwizzle channel_r = mglMTLSwizzleForGLSwizzle(tex, tex->params.swizzle_r);
    MTLTextureSwizzle channel_g = mglMTLSwizzleForGLSwizzle(tex, tex->params.swizzle_g);
    MTLTextureSwizzle channel_b = mglMTLSwizzleForGLSwizzle(tex, tex->params.swizzle_b);
    MTLTextureSwizzle channel_a = mglMTLSwizzleForGLSwizzle(tex, tex->params.swizzle_a);

    tex_desc.swizzle = MTLTextureSwizzleChannelsMake(channel_r, channel_g, channel_b, channel_a);
}

/* mglTextureUploadNeedsSingleChannelSwizzle, mglResolveR8SwizzledComponent,
 * and mglCreateSingleChannelSwizzledUpload now live in mgl_texture_compat.m —
 * see mgl_texture_compat.h. */


- (MGLMetalTextureRef) createMTLTextureFromGLTexture:(Texture *) tex
{
    MGL_ASSERT_GL_THREAD();
    mglMetalCountCreate(MGLMetalKindTexture);
    // PROPER FIX: Enhanced pre-creation validation to prevent AGX driver issues
    if (!_device || !_commandQueue) {
        NSLog(@"MGL ERROR: Metal device or command queue not available for texture creation");
        return nil;
    }

    // Check if we're in a recovery state that would make texture creation futile
    if ([self shouldSkipGPUOperations]) {
        NSLog(@"MGL AGX: GPU operations temporarily suspended during recovery");
        return nil;
    }

    // Validate texture dimensions to prevent Metal assertion failures.
    // Texture buffers (GL_TEXTURE_BUFFER) can have very large widths (millions of texels)
    // since they map to MTLTextureTypeTextureBuffer which uses GPU address space.
    if (tex->target != GL_TEXTURE_BUFFER) {
        if (!tex || tex->width <= 0 || tex->height <= 0 ||
            tex->width > 32768 || tex->height > 32768 || tex->depth > 32768) {
            NSLog(@"MGL ERROR: Invalid texture dimensions %dx%dx%d - rejecting",
                  tex ? tex->width : 0, tex ? tex->height : 0, tex ? tex->depth : 0);
            tex->dirty_bits = 0;
            return nil;
        }
    }

    if (tex->target == GL_TEXTURE_BUFFER) {
        return [self createMTLTexelBufferTexture:tex];
    }

    NSUInteger width, height, depth;

    MTLTextureDescriptor *tex_desc;
    MTLTextureType tex_type;
    MTLPixelFormat pixelFormat;
    uint num_faces;
    GLuint effective_mipmap_levels;
    GLuint upload_level_count;
    BOOL storageMipmapped;
    BOOL mipmapped;
    BOOL is_array;
    BOOL texture1DBackedBy2D;
    BOOL texture1DArrayBackedBy2DArray;

    effective_mipmap_levels = 0;
    upload_level_count = 0;
    storageMipmapped = NO;

    MGLRenderCppTextureTargetPlan targetPlan = {0};
    if (mglRenderCppTextureTargetPlan(
            (uint32_t)tex->target,
            (uint32_t)tex->samples,
            &targetPlan) != 0) {
        NSLog(@"MGL TEXTURE ERROR: unsupported texture target 0x%x for Metal texture creation tex=%u",
              tex->target,
              tex->name);
        return nil;
    }
    tex_type = (MTLTextureType)targetPlan.texture_type;
    num_faces = (uint)targetPlan.num_faces;
    is_array = targetPlan.is_array != 0u;
    texture1DBackedBy2D = targetPlan.texture_1d_backed_by_2d != 0u;
    texture1DArrayBackedBy2DArray =
        targetPlan.texture_1d_array_backed_by_2d_array != 0u;

    if (![self checkTextureCompleteness:tex
                               texType:tex_type
                              numFaces:num_faces
                  effectiveMipmapLevels:&effective_mipmap_levels
                      storageMipmapped:&storageMipmapped]) {
        return nil;
    }

    // PROPER FIX: Get original texture format and validate for AGX compatibility
    pixelFormat = mtlPixelFormatForGLTex(tex);
    BOOL expandsSingleChannelSwizzle = mglTextureUploadNeedsSingleChannelSwizzle(tex);
    if (expandsSingleChannelSwizzle) {
        pixelFormat = MTLPixelFormatRGBA8Unorm;
    }

    // Validate format compatibility with AGX, but preserve original intent
    BOOL needsFormatConversion = NO;
    MTLPixelFormat originalFormat = pixelFormat;

    // Check for AGX-incompatible formats and only convert when necessary
    switch(pixelFormat) {
        case MTLPixelFormatB5G6R5Unorm:
        case MTLPixelFormatBGR5A1Unorm:
        case MTLPixelFormatA1BGR5Unorm:
            // 16-bit formats can cause issues on AGX
            needsFormatConversion = YES;
            pixelFormat = MTLPixelFormatRGBA8Unorm;
            break;
        case MTLPixelFormatPVRTC_RGBA_2BPP:
        case MTLPixelFormatPVRTC_RGBA_4BPP:
        case MTLPixelFormatPVRTC_RGB_2BPP:
        case MTLPixelFormatPVRTC_RGB_4BPP:
            // PVRTC compression can cause issues in virtualization
            needsFormatConversion = YES;
            pixelFormat = MTLPixelFormatRGBA8Unorm;
            break;
        case MTLPixelFormatEAC_R11Unorm:
        case MTLPixelFormatEAC_RG11Unorm:
        case MTLPixelFormatEAC_RGBA8:
        case MTLPixelFormatETC2_RGB8:
        case MTLPixelFormatETC2_RGB8A1:
            // ETC/ETC2 compression can cause issues on AGX
            needsFormatConversion = YES;
            pixelFormat = MTLPixelFormatRGBA8Unorm;
            break;
        default:
            // Most modern formats should work fine
            break;
    }

    /* Metal does not allow depth/stencil pixel formats with MTLTextureType1DArray.
     * Promote to MTLTextureType2DArray with height=1, mirroring how mipmapped
     * 1D array textures are already promoted below.  Without this, creating a
     * GL_TEXTURE_1D_ARRAY depth texture (e.g. sampler_1d_array_shadow) triggers
     * a Metal validation assertion crash. */
    if (tex_type == MTLTextureType1DArray) {
        switch (pixelFormat) {
            case MTLPixelFormatDepth16Unorm:
            case MTLPixelFormatDepth32Float:
            case MTLPixelFormatStencil8:
            case MTLPixelFormatDepth24Unorm_Stencil8:
            case MTLPixelFormatDepth32Float_Stencil8:
            case MTLPixelFormatX32_Stencil8:
            case MTLPixelFormatX24_Stencil8:
                tex_type = MTLTextureType2DArray;
                texture1DArrayBackedBy2DArray = true;
                break;
            default:
                break;
        }
    }

    width = tex->width;
    height = tex->height;
    depth = tex->depth;
    if (tex_type == MTLTextureType2DMultisample ||
        tex_type == MTLTextureType2DMultisampleArray) {
        storageMipmapped = NO;
        effective_mipmap_levels = 1u;
        tex->mipmapped = false;
    }

    mipmapped = storageMipmapped;
    upload_level_count = mipmapped ? effective_mipmap_levels : tex->num_levels;

    tex_desc = [[MTLTextureDescriptor alloc] init];
    tex_desc.textureType = tex_type;
    tex_desc.pixelFormat = pixelFormat;
    tex_desc.width = width;
    tex_desc.height = (tex_type == MTLTextureType1D ||
                       tex_type == MTLTextureType1DArray) ? 1 : height;
    if (tex_type == MTLTextureType2DMultisample ||
        tex_type == MTLTextureType2DMultisampleArray) {
        /* Metal only supports a device-specific subset of sample counts.
         * Apple Silicon (e.g. M4) only supports 1/2/4 — NOT 8.  Delegate to
         * the AGX Capability Layer for centralized clamping. */
        NSUInteger samples = MAX((NSUInteger)2u, (NSUInteger)tex->samples);
        samples = MGLCapabilityClampSampleCount(&_capability, samples);
        tex_desc.sampleCount = samples;
    }

    // CONSERVATIVE: Use only Metal API patterns that work reliably with AGX driver
    tex_desc.cpuCacheMode = MGLCapabilityUseConservativeCPUCache(&_capability)
        ? MTLCPUCacheModeWriteCombined
        : MTLCPUCacheModeDefaultCache;

    // Use shared storage for textures that need CPU upload (blit/replaceRegion).
    // Private storage is only safe for pure GPU render targets on Apple Silicon.
    bool hasUploadableCPUData = mglTextureHasUploadableCPUData(tex, num_faces, upload_level_count);
    bool needsCpuUpload = ((tex->dirty_bits & DIRTY_TEXTURE_DATA) != 0) && hasUploadableCPUData;
    tex_desc.storageMode = needsCpuUpload ? MTLStorageModeShared : MTLStorageModePrivate;

    // Normalize depth/array semantics per Metal texture type.
    if (tex_type == MTLTextureTypeCube) {
        if (width != height) {
            NSLog(@"MGL ERROR: invalid cube texture size %lux%lu for tex=%u glTarget=0x%x",
                  (unsigned long)width, (unsigned long)height, tex->name, tex->target);
        }
        tex_desc.depth = 1;
    } else if (tex_type == MTLTextureTypeCubeArray) {
        if (width != height) {
            NSLog(@"MGL ERROR: invalid cube-array texture size %lux%lu for tex=%u glTarget=0x%x",
                  (unsigned long)width, (unsigned long)height, tex->name, tex->target);
        }

        // GL cube-map-array depth is usually layer count (faces), so convert to cube count.
        // If depth is already cube-count (non-multiple of 6), keep it as-is.
        NSUInteger cubeCount = depth;
        if (cubeCount >= 6 && (cubeCount % 6) == 0) {
            cubeCount = cubeCount / 6;
        } else if (cubeCount > 1 && (cubeCount % 6) != 0) {
            NSLog(@"MGL WARNING: cube-array depth=%lu is not a multiple of 6, treating as cube count",
                  (unsigned long)cubeCount);
        }

        tex_desc.arrayLength = MAX((NSUInteger)1, cubeCount);
        tex_desc.depth = 1;
    } else if (tex_type == MTLTextureType1DArray) {
        tex_desc.arrayLength = MAX((NSUInteger)1, height);
        tex_desc.depth = 1;
    } else if (is_array) {
        tex_desc.arrayLength = MAX((NSUInteger)1, depth);
        tex_desc.depth = 1;
    } else {
        /* For 3D and other non-array textures, arrayLength must be 1.
         * Some Metal drivers report getNumSlices()==0 when arrayLength
         * is left at its default, causing "slice OOB" assertions. */
        tex_desc.arrayLength = 1;
        tex_desc.depth = MAX((NSUInteger)1, depth);
    }

    if (mipmapped)
    {
        if (tex_type == MTLTextureType1D) {
            tex_type = MTLTextureType2D;
            texture1DBackedBy2D = true;
        }
        /* Metal does not allow mipmapLevelCount > 1 for MTLTextureType1DArray.
         * Promote to MTLTextureType2DArray with height=1 to support mipmapped
         * 1D array textures.  The upload code checks texture1DArrayBackedBy2DArray
         * to treat each slice as 1 pixel tall. */
        if (tex_type == MTLTextureType1DArray) {
            tex_type = MTLTextureType2DArray;
            texture1DArrayBackedBy2DArray = true;
        }
        tex_desc.mipmapLevelCount = MAX((GLuint)1, effective_mipmap_levels);
    }

    if (texture1DBackedBy2D) {
        tex_desc.textureType = MTLTextureType2D;
        tex_desc.height = 1;
    }
    if (texture1DArrayBackedBy2DArray) {
        tex_desc.textureType = MTLTextureType2DArray;
        /* For GL_TEXTURE_1D_ARRAY, the GL height parameter is the array slice
         * count.  Since tex_type was promoted to MTLTextureType2DArray above,
         * the arrayLength branch at line ~12397 (keyed on MTLTextureType1DArray)
         * was skipped, leaving arrayLength=1 from the is_array/depth fallback.
         * Set arrayLength from the GL height (slice count) here. */
        tex_desc.arrayLength = MAX((NSUInteger)1, height);
        tex_desc.height = 1;
    }

    /* GL image access mode (GL_READ_ONLY / GL_WRITE_ONLY / GL_READ_WRITE)
     * only governs the image binding, NOT the texture's overall capabilities.
     * A texture bound as a write-only image may still be sampled from via
     * sampler2D in the same shader.  Metal requires MTLTextureUsageShaderRead
     * for sampling, so always include it alongside the image write flag. */
    switch(tex->access)
    {
        case GL_READ_ONLY:
            tex_desc.usage = MTLTextureUsageShaderRead; break;
        case GL_WRITE_ONLY:
            tex_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; break;
        case GL_READ_WRITE:
            tex_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; break;
        default:
            NSLog(@"MGL TEXTURE ERROR: invalid texture access 0x%x for tex=%u",
                  tex->access,
                  tex->name);
            return nil;
    }

    if (tex->is_render_target)
    {
        tex_desc.usage |= MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    }

    // Allow safe same-memory format reinterpretation (e.g. RGBA8 <-> BGRA8)
    // for blit/present paths where OpenGL attachments and drawable formats differ.
    tex_desc.usage |= MTLTextureUsagePixelFormatView;

    if (tex_desc.textureType == MTLTextureTypeCube || tex_desc.textureType == MTLTextureTypeCubeArray) {
        NSLog(@"MGL CUBE DESC tex=%u glTarget=0x%x type=%lu width=%lu height=%lu depth=%lu arrayLength=%lu pixelFormat=%lu usage=%lu storage=%lu mipmapped=%d",
              tex->name,
              tex->target,
              (unsigned long)tex_desc.textureType,
              (unsigned long)tex_desc.width,
              (unsigned long)tex_desc.height,
              (unsigned long)tex_desc.depth,
              (unsigned long)tex_desc.arrayLength,
              (unsigned long)tex_desc.pixelFormat,
              (unsigned long)tex_desc.usage,
              (unsigned long)tex_desc.storageMode,
              (int)mipmapped);
    }

    // CRITICAL FIX: Proper validation instead of assertions
    if (!tex_desc) {
        NSLog(@"MGL ERROR: Failed to create texture descriptor");
        return NULL;
    }

    if (tex->params.swizzled && !expandsSingleChannelSwizzle)
    {
        [self swizzleTexDesc:tex_desc forTex:tex];
    }

    MGLMetalTextureRef texture;

    // CRITICAL FIX: Safe texture creation with proper validation
    @try {
        texture = mglTextureCreateTexture(_device, tex_desc);
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception creating texture: %@", exception);
        [self recordGPUError];
        return NULL;
    }

    // CRITICAL FIX: Validate texture creation result instead of asserting
    if (!texture) {
        NSLog(@"MGL ERROR: Failed to create Metal texture with descriptor");
        return NULL;
    }

    BOOL cpuUploadRequired =
        ((tex->dirty_bits & DIRTY_TEXTURE_DATA) != 0) && hasUploadableCPUData;
    BOOL cpuUploadVerified = !cpuUploadRequired;
    BOOL allLevelsUploaded = YES;

    if (cpuUploadRequired)
    {
        if (![self uploadDirtyCPUTextureData:tex
                                       metal:texture
                                 pixelFormat:pixelFormat
                                   numFaces:num_faces
                           uploadLevelCount:upload_level_count
                                    isArray:is_array
                         texture1DBackedBy2D:texture1DBackedBy2D
                   texture1DArrayBackedBy2DArray:texture1DArrayBackedBy2DArray
                                    texType:tex_type
                        outAllLevelsUploaded:&allLevelsUploaded]) {
            return nil;
        }
    }
    else
    {
        if (hasUploadableCPUData) {
            [self reUploadExistingCPUTextureData:tex
                                            metal:texture
                                      pixelFormat:pixelFormat
                                        numFaces:num_faces
                                uploadLevelCount:upload_level_count
                                          isArray:is_array
                               texture1DBackedBy2D:texture1DBackedBy2D
                         texture1DArrayBackedBy2DArray:texture1DArrayBackedBy2DArray
                                             texType:tex_type];
        } else if (tex->is_render_target || mglMetalPixelFormatIsDepthOrStencil(pixelFormat)) {
            static uint64_t s_skipRenderTargetFillLogs = 0;
            uint64_t hit = ++s_skipRenderTargetFillLogs;
            if (hit <= 8ull || (hit % 2048ull) == 0ull) {
                NSLog(@"MGL TEXTURE SKIP implicit fill tex=%u renderTarget=%u format=%lu sourceSafe=0 hit=%llu",
                      (unsigned)tex->name,
                      (unsigned)tex->is_render_target,
                      (unsigned long)pixelFormat,
                      (unsigned long long)hit);
            }
        } else {
            [self fillTextureWithSafeInitialContents:texture
                                                 tex:tex
                                         pixelFormat:pixelFormat];
        }
    }

    if (cpuUploadRequired && tex->target == GL_TEXTURE_2D && texture.textureType == MTLTextureType2D) {
        BOOL fullCPUUploadVerified = [self uploadFullCPUTextureDataIntoTexture:tex
                                                                           metal:texture
                                                                          reason:"createMTLTexture.cpuData"];
        cpuUploadVerified = allLevelsUploaded && fullCPUUploadVerified;
    } else if (cpuUploadRequired) {
        /*
         * Non-2D uploads still use the legacy creation path above. The current GUI
         * atlas failure is 2D; avoid changing array/cube semantics in this pass.
         * If any mip level was skipped (invalid layout, NULL data, etc.) keep
         * DIRTY_TEXTURE_DATA set so the level gets retried on next bind.
         */
        cpuUploadVerified = allLevelsUploaded;
    }

    if (cpuUploadRequired && !cpuUploadVerified) {
        static uint64_t s_createTextureCPUUploadIncompleteLogs = 0;
        uint64_t hit = ++s_createTextureCPUUploadIncompleteLogs;
        if (hit <= 8ull || (hit % 2048ull) == 0ull) {
            TextureLevel *level0 = mglTraceTextureBaseLevel(tex);
            NSLog(@"MGL TEXTURE CREATE CPU-UPLOAD INCOMPLETE tex=%u target=0x%x dirtyBefore=0x%x level0=%ux%u source=%u upload=%lu hit=%llu",
                  (unsigned)tex->name,
                  (unsigned)tex->target,
                  (unsigned)tex->dirty_bits,
                  level0 ? (unsigned)level0->width : 0u,
                  level0 ? (unsigned)level0->height : 0u,
                  level0 ? (unsigned)level0->last_init_source : 0u,
                  (unsigned long)(level0 ? level0->last_upload_size : 0u),
                  (unsigned long long)hit);
        }
        tex->dirty_bits &= ~(DIRTY_TEXTURE_LEVEL | DIRTY_TEXTURE_ACCESS);
        tex->dirty_bits |= DIRTY_TEXTURE_DATA;
    } else {
        tex->dirty_bits = 0;
    }

    [self logMTLTextureMipDiagnostics:tex metal:texture effectiveMipLevels:effective_mipmap_levels];

    [self recordGPUSuccess];

    return texture;
}

- (MGLMetalTextureRef)createMTLTexelBufferTexture:(Texture *)tex
{
    Buffer *sourceBuffer = tex->texture_buffer;
    if (!sourceBuffer || tex->texture_buffer_size <= 0) {
        NSLog(@"MGL TEXBUFFER ERROR: tex=%u has no attached buffer/size buffer=%p size=%lld",
              tex->name,
              sourceBuffer,
              (long long)tex->texture_buffer_size);
        return nil;
    }

    if (tex->texture_buffer_offset < 0 ||
        tex->texture_buffer_offset > sourceBuffer->size ||
        tex->texture_buffer_size > sourceBuffer->size - tex->texture_buffer_offset) {
        NSLog(@"MGL TEXBUFFER ERROR: invalid range tex=%u buffer=%u off=%lld size=%lld bufferSize=%lld",
              tex->name,
              sourceBuffer->name,
              (long long)tex->texture_buffer_offset,
              (long long)tex->texture_buffer_size,
              (long long)sourceBuffer->size);
        return nil;
    }

    NSUInteger bytesPerTexel = [self bytesPerPixelForFormat:tex->internalformat];
    if (bytesPerTexel == 0) {
        NSLog(@"MGL TEXBUFFER ERROR: unsupported internal format 0x%x tex=%u buffer=%u",
              tex->internalformat,
              tex->name,
              sourceBuffer->name);
        return nil;
    }

    NSUInteger texelCount = (NSUInteger)tex->texture_buffer_size / bytesPerTexel;
    if (texelCount == 0) {
        NSLog(@"MGL TEXBUFFER ERROR: zero texel count tex=%u buffer=%u size=%lld bpt=%lu",
              tex->name,
              sourceBuffer->name,
              (long long)tex->texture_buffer_size,
              (unsigned long)bytesPerTexel);
        return nil;
    }

    MTLPixelFormat bufferPixelFormat = (tex->internalformat == GL_RGBA8)
        ? MTLPixelFormatRGBA8Uint
        : mtlPixelFormatForGLTex(tex);
    if (bufferPixelFormat == MTLPixelFormatInvalid || bufferPixelFormat == 0) {
        NSLog(@"MGL TEXBUFFER ERROR: invalid Metal format for tex=%u internal=0x%x",
              tex->name,
              tex->internalformat);
        return nil;
    }

    if (![self processBuffer:sourceBuffer]) {
        NSLog(@"MGL TEXBUFFER ERROR: failed to process source buffer tex=%u buffer=%u",
              tex->name,
              sourceBuffer->name);
        return nil;
    }

    const uint8_t *sourceBytes = NULL;
    if (sourceBuffer->data.buffer_data) {
        sourceBytes = ((const uint8_t *)(uintptr_t)sourceBuffer->data.buffer_data) + (size_t)tex->texture_buffer_offset;
    } else if (sourceBuffer->data.mtl_data) {
        MGLMetalBufferRef mtlBuffer = (__bridge MGLMetalBufferRef)(sourceBuffer->data.mtl_data);
        if (mtlBuffer && mtlBuffer.contents) {
            sourceBytes = ((const uint8_t *)mtlBuffer.contents) + (size_t)tex->texture_buffer_offset;
        }
    }

    if (!sourceBytes) {
        NSLog(@"MGL TEXBUFFER ERROR: no readable backing for tex=%u buffer=%u cpu=%p mtl=%p",
              tex->name,
              sourceBuffer->name,
              (void *)(uintptr_t)sourceBuffer->data.buffer_data,
              sourceBuffer->data.mtl_data);
        return nil;
    }

    // The AIR backend emits Minecraft's CloudFaces texel buffer as a
    // texture2d<int>. Keep GL lookup semantics as GL_TEXTURE_BUFFER, but
    // create a Metal 2D backing so the generated MSL argument type matches.
    // A texel buffer can be much wider than Metal's max 2D texture width,
    // so pack it into rows instead of creating texelCount x 1.
    /*
     * The AIR backend lowers GL texture buffers to 2D Metal textures and emits
     * spvTexelBufferCoord(tc) using its MSL texel_buffer_texture_width
     * option. Keep this packing width in lockstep with program.c.
     */
    static const NSUInteger kMGLTexelBufferTextureWidth = 4096u;
    NSUInteger max2DSize = (NSUInteger)MIN((GLuint)kMGLTexelBufferTextureWidth,
                                           ctx ? MGL_STATE(ctx)->var.max_texture_size : (GLuint)kMGLTexelBufferTextureWidth);
    if (max2DSize == 0 || max2DSize > kMGLTexelBufferTextureWidth) {
        max2DSize = kMGLTexelBufferTextureWidth;
    }

    NSUInteger texWidth = MIN(texelCount, max2DSize);
    NSUInteger texHeight = (texelCount + texWidth - 1) / texWidth;
    if (texHeight == 0 || texHeight > max2DSize) {
        NSLog(@"MGL TEXBUFFER ERROR: texel buffer too large for 2D fallback tex=%u buffer=%u texels=%lu packed=%lux%lu max=%lu",
              tex->name,
              sourceBuffer->name,
              (unsigned long)texelCount,
              (unsigned long)texWidth,
              (unsigned long)texHeight,
              (unsigned long)max2DSize);
        return nil;
    }

    NSUInteger bytesPerRow = texWidth * bytesPerTexel;
    NSUInteger packedBytes = bytesPerRow * texHeight;
    NSMutableData *packedData = nil;
    const uint8_t *uploadBytes = sourceBytes;

    /* Channel expansion for 3-channel RGB -> 4-channel RGBA Metal formats.
     * GL_RGB32* (12 bytes/texel) maps to Metal RGBA32* (16 bytes/texel).
     * Expand each texel by inserting a default alpha before uploading. */
    NSMutableData *expandedData = nil;
    if (mglTextureNeedsChannelExpansion(tex->internalformat, bufferPixelFormat)) {
        NSUInteger srcCompBytes = 0;
        NSUInteger dstCompBytes = 0;
        uint64_t alphaDefault = 0;
        switch (bufferPixelFormat) {
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
                break;
        }
        if (srcCompBytes > 0) {
            NSUInteger srcPixelBytes = srcCompBytes * 3;
            NSUInteger dstPixelBytes = dstCompBytes * 4;
            NSUInteger expandedBytesPerRow = texWidth * dstPixelBytes;
            NSUInteger expandedPackedBytes = expandedBytesPerRow * texHeight;
            expandedData = [NSMutableData dataWithLength:expandedPackedBytes];
            if (expandedData && expandedData.mutableBytes) {
                /* P4.4: 通道扩展在 C++（mglRenderCppTextureExpandRGBToRGBA，
                 * 纯数据变换，两门共用）。dst 由 NSMutableData 持有，生命周期
                 * 仍 ARC 管理；与内联版逐 texel 等价。 */
                if (mglRenderCppTextureExpandRGBToRGBA(
                        sourceBytes, expandedData.mutableBytes, texelCount,
                        texWidth, texHeight, srcCompBytes, dstCompBytes,
                        alphaDefault) != 0) {
                    NSLog(@"MGL TEXBUFFER ERROR: channel expansion failed tex=%u buffer=%u",
                          tex->name,
                          sourceBuffer->name);
                    return nil;
                }
                uploadBytes = (const uint8_t *)expandedData.bytes;
                bytesPerRow = expandedBytesPerRow;
                packedBytes = expandedPackedBytes;
            }
        }
    }

    if (texHeight > 1 && !expandedData) {
        packedData = [NSMutableData dataWithLength:packedBytes];
        if (!packedData || !packedData.mutableBytes) {
            NSLog(@"MGL TEXBUFFER ERROR: failed allocating packed data tex=%u buffer=%u bytes=%lu",
                  tex->name,
                  sourceBuffer->name,
                  (unsigned long)packedBytes);
            return nil;
        }

        memcpy(packedData.mutableBytes, sourceBytes, (size_t)tex->texture_buffer_size);
        uploadBytes = (const uint8_t *)packedData.bytes;
    }

    uint64_t sourceHash = mglTraceHashBytes(sourceBytes, (size_t)tex->texture_buffer_size);
    uint64_t uploadHash = mglTraceHashBytes(uploadBytes, packedBytes);
    char sourceHead[64];
    char uploadHead[64];
    sourceHead[0] = '\0';
    uploadHead[0] = '\0';
    mglTraceFormatBytes(sourceBytes, (size_t)MIN((NSUInteger)tex->texture_buffer_size, (NSUInteger)64), sourceHead, sizeof(sourceHead));
    mglTraceFormatBytes(uploadBytes, (size_t)MIN(packedBytes, (NSUInteger)64), uploadHead, sizeof(uploadHead));

    MTLTextureDescriptor *bufferDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:bufferPixelFormat
                                                           width:texWidth
                                                          height:texHeight
                                                       mipmapped:NO];
    bufferDesc.usage = MTLTextureUsageShaderRead;
    bufferDesc.storageMode = MTLStorageModeShared;

    MGLMetalTextureRef bufferTexture = nil;
    @try {
        bufferTexture = mglTextureCreateTexture(_device, bufferDesc);
        if (bufferTexture) {
            mglTextureReplaceRegion(
                bufferTexture, MTLRegionMake2D(0, 0, texWidth, texHeight),
                0, 0, uploadBytes, bytesPerRow, 0, NO);
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL TEXBUFFER ERROR: failed creating/uploading tex=%u buffer=%u exception=%@",
              tex->name,
              sourceBuffer->name,
              exception);
        return nil;
    }

    if (!bufferTexture) {
        NSLog(@"MGL TEXBUFFER ERROR: Metal texture creation returned nil tex=%u buffer=%u format=%lu texels=%lu",
              tex->name,
              sourceBuffer->name,
              (unsigned long)bufferPixelFormat,
              (unsigned long)texelCount);
        return nil;
    }

    tex->dirty_bits = 0;
    sourceBuffer->data.dirty_bits = 0;

    NSMutableData *readbackData = [NSMutableData dataWithLength:packedBytes];
    uint64_t readbackHash = 0ull;
    char readbackHead[64];
    readbackHead[0] = '\0';
    if (readbackData.mutableBytes) {
        mglTextureGetBytes(
            bufferTexture, readbackData.mutableBytes, bytesPerRow, 0,
            MTLRegionMake2D(0, 0, texWidth, texHeight), 0, 0, NO);
        readbackHash = mglTraceHashBytes(readbackData.bytes, packedBytes);
        mglTraceFormatBytes(readbackData.bytes, (size_t)MIN(packedBytes, (NSUInteger)64), readbackHead, sizeof(readbackHead));
    }

    {
        static uint64_t s_texBufferCreateLogs = 0;
        uint64_t hit = ++s_texBufferCreateLogs;
        if (hit <= 2ull || (hit % 4096ull) == 0ull) {
            NSLog(@"MGL TEXBUFFER CREATE tex=%u buffer=%u internal=0x%x mtlFormat=%lu texels=%lu packed=%lux%lu rowBytes=%lu bytes=%lld offset=%lld as=texture2d sourceHash=0x%016llx uploadHash=0x%016llx readbackHash=0x%016llx sourceHead=%s uploadHead=%s readbackHead=%s",
                  tex->name,
                  sourceBuffer->name,
                  tex->internalformat,
                  (unsigned long)bufferPixelFormat,
                  (unsigned long)texelCount,
                  (unsigned long)texWidth,
                  (unsigned long)texHeight,
                  (unsigned long)bytesPerRow,
                  (long long)tex->texture_buffer_size,
                  (long long)tex->texture_buffer_offset,
                  (unsigned long long)sourceHash,
                  (unsigned long long)uploadHash,
                  (unsigned long long)readbackHash,
                  sourceHead,
                  uploadHead,
                  readbackHead);
        }
    }

    [self recordGPUSuccess];
    return bufferTexture;
}

- (BOOL)checkTextureCompleteness:(Texture *)tex
                          texType:(MTLTextureType)tex_type
                         numFaces:(uint)num_faces
             effectiveMipmapLevels:(GLuint *)outEffectiveMipmapLevels
                 storageMipmapped:(BOOL *)outStorageMipmapped
{
    (void)tex_type;  /* unused: completeness does not depend on Metal texture type */
    GLuint effective_mipmap_levels = tex->mipmap_levels;
    BOOL storageMipmapped = NO;

    uint completeness_check_faces = (tex->target == GL_TEXTURE_CUBE_MAP_ARRAY) ? 1 : num_faces;

    /* Texture storage is independent from GL_TEXTURE_MAX_LEVEL.  Minecraft
     * uses BASE/MAX_LEVEL to express temporary GpuTextureView mip windows; if
     * those sampler parameters shrink the Metal texture allocation, later
     * full-atlas sampling loses the higher mip levels and distant terrain
     * reads empty/incorrect data.  Apply BASE/MAX only to completeness checks
     * and sampled Metal views, not to the underlying storage level count. */

    /* For CUBE_MAP_ARRAY, glTexImage3D stores all layer data in faces[0] with
     * depth = 6 * num_cubes.  Faces 1-5 are never populated by createTextureLevel,
     * so only check face 0 for completeness.  The upload code also reads from
     * face 0 and distributes slices to Metal array layers. */

    storageMipmapped = (tex->mipmap_levels > 1u) &&
        (tex->num_levels > 1u || tex->is_render_target);

    if (tex->num_levels > 1)
    {
        // mipmapped texture
        if (effective_mipmap_levels == 0) {
            effective_mipmap_levels = tex->num_levels;
        }

        if (!tex->is_render_target && tex->num_levels < effective_mipmap_levels)
        {
            static uint64_t s_mipmap_count_mismatch_logs = 0;
            if (++s_mipmap_count_mismatch_logs <= 8 || (s_mipmap_count_mismatch_logs % 2048) == 0) {
                NSLog(@"MGL TEXTURE MIP COMPAT: tex=%u target=0x%x size=%ux%u num_levels=%u mipmap_levels=%u effective=%u base=%u max=%u immutable=%u isRT=%u; capping Metal mip count to uploaded levels hit=%llu",
                      tex->name,
                      tex->target,
                      tex->width,
                      tex->height,
                      tex->num_levels,
                      tex->mipmap_levels,
                      effective_mipmap_levels,
                      tex->params.base_level,
                      tex->params.max_level,
                      tex->immutable_storage,
                      tex->is_render_target,
                      (unsigned long long)s_mipmap_count_mismatch_logs);
            }
            effective_mipmap_levels = tex->num_levels;
        }

        /* GL texture completeness only requires levels in
         * [base_level, min(max_level, mipmap_levels-1)] to be complete.
         * Levels below base_level may be uninitialised and must NOT cause
         * the texture to be rejected.  Minecraft 1.21.11 sets base_level>0
         * on mipmap texture views (GlCommandEncoder.java). */
        GLuint check_start = tex->params.base_level;
        GLuint check_end = (tex->params.max_level == 1000u)
            ? (tex->mipmap_levels > 0u ? tex->mipmap_levels - 1u : 0u)
            : tex->params.max_level;
        if (check_end >= tex->mipmap_levels)
            check_end = (tex->mipmap_levels > 0u) ? tex->mipmap_levels - 1u : 0u;
        if (check_end < check_start)
            check_end = check_start;

        for(int face=0; face<completeness_check_faces; face++)
        {
            for (GLuint i=check_start; i<=check_end; i++)
            {
                // incomplete texture
                if (tex->faces[face].levels[i].complete == false) {
                    static uint64_t s_incomplete_mip_logs = 0;
                    if (++s_incomplete_mip_logs <= 32 || (s_incomplete_mip_logs % 512) == 0) {
                        NSLog(@"MGL TEXTURE INCOMPLETE: tex=%u target=0x%x face=%d level=%u incomplete num_levels=%u mipmap_levels=%u effective=%u base=%u max=%u check=[%u,%u] hit=%llu",
                              tex->name,
                              tex->target,
                              face,
                              i,
                              tex->num_levels,
                              tex->mipmap_levels,
                              effective_mipmap_levels,
                              tex->params.base_level,
                              tex->params.max_level,
                              check_start,
                              check_end,
                              (unsigned long long)s_incomplete_mip_logs);
                    }
                    return NO;
                }
            }
        }

        tex->mipmapped = true;
    }
    else if (tex->num_levels == 1)
    {
        if (!storageMipmapped) {
            effective_mipmap_levels = 1;
        }
        // single level texture
        // incomplete texture
        for(int face=0; face<completeness_check_faces; face++)
        {
            if (tex->faces[face].levels[0].complete == false)
            {
                static uint64_t s_incomplete_base_logs = 0;
                if (++s_incomplete_base_logs <= 32 || (s_incomplete_base_logs % 512) == 0) {
                    NSLog(@"MGL TEXTURE INCOMPLETE: tex=%u target=0x%x face=%d base incomplete size=%ux%u hit=%llu",
                          tex->name,
                          tex->target,
                          face,
                          tex->width,
                          tex->height,
                          (unsigned long long)s_incomplete_base_logs);
                }
                return NO;
            }
        }
    }
    else
    {
        NSLog(@"MGL TEXTURE ERROR: texture %u has no complete levels for Metal creation target=0x%x",
              tex->name,
              tex->target);
        return NO;
    }

    tex->complete = true;

    if (outEffectiveMipmapLevels) *outEffectiveMipmapLevels = effective_mipmap_levels;
    if (outStorageMipmapped) *outStorageMipmapped = storageMipmapped;
    return YES;
}

- (void)logMTLTextureMipDiagnostics:(Texture *)tex
                              metal:(MGLMetalTextureRef)texture
               effectiveMipLevels:(GLuint)effective_mipmap_levels
{
    static uint64_t s_mipDiagLogs = 0;
    uint64_t diagHit = ++s_mipDiagLogs;
    if (kMGLDiagnosticStateLogs &&
        (diagHit <= 128ull || (diagHit % 512ull) == 0ull)) {
        NSUInteger mtlMipCount = texture.mipmapLevelCount;
        MTLPixelFormat mtlFmt = texture.pixelFormat;
        MTLStorageMode mtlStorage = texture.storageMode;
        NSUInteger uploadedLevels = 0;
        NSUInteger skippedLevels = 0;
        NSUInteger skippedSourceNone = 0;
        NSUInteger skippedNoData = 0;
        NSMutableString *levelSummary = [NSMutableString stringWithCapacity:256];
        NSUInteger levelsToSummarize = MIN((NSUInteger)tex->num_levels, (NSUInteger)16);
        for (NSUInteger lvl = 0; lvl < levelsToSummarize; lvl++) {
            TextureLevel *tl = (tex->faces[0].levels && lvl < tex->num_levels)
                ? &tex->faces[0].levels[lvl] : NULL;
            if (!tl) { [levelSummary appendString:@"-"]; continue; }
            bool uploadable = mglTextureLevelHasUploadableCPUData(tl);
            if (uploadable) uploadedLevels++; else skippedLevels++;
            if (!uploadable) {
                if (tl->last_init_source == kTexImageNull || tl->last_init_source == kTexInitNone)
                    skippedSourceNone++;
                if (!tl->has_initialized_data && !tl->ever_written)
                    skippedNoData++;
            }
            [levelSummary appendFormat:@"[%u:s%u:w%u:e%u:i%u]",
                (unsigned)lvl, (unsigned)tl->last_init_source,
                (unsigned)tl->width, (unsigned)tl->ever_written,
                (unsigned)tl->has_initialized_data];
        }
        mglTraceLogNSString(@"MGL TEX_MIP_DIAG tex=%u target=0x%x dims=%ux%u internal=0x%x "
                      @"numLevels=%u mipmapLevels=%u effectiveMipLevels=%u mtlMipCount=%lu "
                      @"mtlFmt=%lu mtlStorage=%ld mipmapped=%d baseLevel=%u maxLevel=%u "
                      @"uploadedLevels=%lu skippedLevels=%lu skippedSourceNone=%lu skippedNoData=%lu "
                      @"levels=%@ hit=%llu",
                      (unsigned)tex->name, (unsigned)tex->target,
                      (unsigned)tex->width, (unsigned)tex->height,
                      (unsigned)tex->internalformat,
                      (unsigned)tex->num_levels, (unsigned)tex->mipmap_levels,
                      (unsigned)effective_mipmap_levels, (unsigned long)mtlMipCount,
                      (unsigned long)mtlFmt, (long)mtlStorage, (int)(tex->mipmapped ? 1 : 0),
                      (unsigned)tex->params.base_level, (unsigned)tex->params.max_level,
                      (unsigned long)uploadedLevels, (unsigned long)skippedLevels,
                      (unsigned long)skippedSourceNone, (unsigned long)skippedNoData,
                      levelSummary, (unsigned long long)diagHit);
    }
}

// AGX-SAFE Fallback texture creation for GPU error recovery scenarios
- (MGLMetalTextureRef) createFallbackMTLTexture:(Texture *) tex
{
    // Validate texture parameters before creating Metal texture to prevent Metal assertion failures
    if (!tex || tex->width <= 0 || tex->height <= 0 || tex->width > 32768 || tex->height > 32768) {
        NSLog(@"MGL AGX: Skipping fallback texture creation - invalid dimensions %dx%d",
              tex ? tex->width : 0, tex ? tex->height : 0);
        return nil;
    }

    NSLog(@"MGL AGX: Creating emergency fallback texture (size: %dx%dx%d)", tex->width, tex->height, tex->depth);

    @try {
        MTLPixelFormat fallbackFormat = mtlPixelFormatForGLTex(tex);
        if (fallbackFormat == MTLPixelFormatInvalid) {
            // Conservative defaults by GL intent when translation is unavailable.
            if (tex->internalformat == GL_DEPTH24_STENCIL8 ||
                tex->internalformat == GL_DEPTH32F_STENCIL8) {
                fallbackFormat = MTLPixelFormatDepth32Float_Stencil8;
            } else if (tex->internalformat == GL_DEPTH_COMPONENT ||
                       tex->internalformat == GL_DEPTH_COMPONENT16 ||
                       tex->internalformat == GL_DEPTH_COMPONENT24 ||
                       tex->internalformat == GL_DEPTH_COMPONENT32 ||
                       tex->internalformat == GL_DEPTH_COMPONENT32F) {
                fallbackFormat = MTLPixelFormatDepth32Float;
            } else {
                fallbackFormat = MTLPixelFormatRGBA8Unorm;
            }
        }

        BOOL isDepthOrStencilFormat =
            (fallbackFormat == MTLPixelFormatDepth16Unorm ||
             fallbackFormat == MTLPixelFormatDepth32Float ||
             fallbackFormat == MTLPixelFormatDepth24Unorm_Stencil8 ||
             fallbackFormat == MTLPixelFormatDepth32Float_Stencil8 ||
             fallbackFormat == MTLPixelFormatStencil8);

        MTLTextureDescriptor *fallbackDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fallbackFormat
                                                                                                    width:MAX(tex->width, 1)
                                                                                                   height:MAX(tex->height, 1)
                                                                                                mipmapped:NO];
        fallbackDesc.usage = MTLTextureUsageShaderRead;
        if (tex->is_render_target || isDepthOrStencilFormat) {
            fallbackDesc.usage |= MTLTextureUsageRenderTarget;
        }
        fallbackDesc.storageMode = MTLStorageModeShared;

        MGLMetalTextureRef fallbackTexture =
            mglTextureCreateTexture(_device, fallbackDesc);

        if (fallbackTexture) {
            // Fill with simple gradient pattern using a simple approach
            NSUInteger width = fallbackTexture.width;
            NSUInteger height = fallbackTexture.height;

            if (!isDepthOrStencilFormat && width <= 512 && height <= 512) {
                uint32_t *gradientData = calloc(width * height, sizeof(uint32_t));
                if (gradientData) {
                    // Create simple red-blue gradient
                    for (NSUInteger y = 0; y < height; y++) {
                        for (NSUInteger x = 0; x < width; x++) {
                            NSUInteger index = y * width + x;
                            uint8_t r = (uint8_t)((x * 255) / width);
                            uint8_t g = 128;
                            uint8_t b = (uint8_t)((y * 255) / height);
                            uint8_t a = 255;
                            gradientData[index] = ((uint32_t)a << 24) | ((uint32_t)b << 16) | ((uint32_t)g << 8) | (uint32_t)r;
                        }
                    }

                    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
                    mglTextureReplaceRegion(
                        fallbackTexture, region, 0, 0, gradientData,
                        width * sizeof(uint32_t), 0, NO);

                    free(gradientData);
                    NSLog(@"MGL AGX: Fallback color texture created with gradient pattern");
                }
            }
        }

        return fallbackTexture;

    } @catch (NSException *exception) {
        NSLog(@"MGL AGX: Even fallback texture creation failed: %@", exception.reason);
        return nil;
    }
}

// Helper function to calculate bytes per pixel for different OpenGL formats
- (NSUInteger)bytesPerPixelForFormat:(GLenum)internalformat
{
    switch(internalformat) {
        case GL_RED:
        case GL_R8:
        case GL_R8I:
        case GL_R8UI:
            return 1;

        case GL_RG:
        case GL_RG8:
        case GL_RG8I:
        case GL_RG8UI:
        case GL_R16:
        case GL_R16F:
        case GL_R16I:
        case GL_R16UI:
            return 2;

        case GL_RGB:
        case GL_RGB8:
        case GL_RGB8I:
        case GL_RGB8UI:
        case GL_SRGB8:
        case GL_R11F_G11F_B10F:
        case GL_RGB9_E5:
            return 3;

        case GL_RGBA:
        case GL_RGBA8:
        case GL_RGBA8I:
        case GL_RGBA8UI:
        case GL_RGB10_A2:
        case GL_RGB10_A2UI:
        case GL_SRGB8_ALPHA8:
        case GL_RG16I:
        case GL_RG16UI:
        case GL_R32I:
        case GL_R32UI:
        case GL_R32F:
            return 4;

        case GL_RGBA16:
        case GL_RGBA16F:
        case GL_RG32I:
        case GL_RG32UI:
        case GL_RG32F:
            return 8;

        case GL_RGB16:
        case GL_RGB16F:
            return 6;

        case GL_RGBA16I:
        case GL_RGBA16UI:
            return 8;

        case GL_RGB32F:
        case GL_RGB32I:
        case GL_RGB32UI:
            return 12;

        case GL_RGBA32F:
        case GL_RGBA32I:
        case GL_RGBA32UI:
            return 16;

        default:
            // Default to 4 bytes for unknown formats
            NSLog(@"MGL WARNING: Unknown internal format 0x%x, defaulting to 4 bytes per pixel", internalformat);
            return 4;
    }
}

- (MGLMetalSamplerStateRef) createMTLSamplerForTexParam:(TextureParameter *)tex_param target:(GLuint)target
{
    mglMetalCountCreate(MGLMetalKindSampler);
    void *sampler = NULL;
    char error[256] = {0};
    if (mglRenderCppCreateSamplerForGL(
            tex_param, target, &sampler, error, sizeof(error)) == 0 &&
        sampler) {
        return (__bridge_transfer MGLMetalSamplerStateRef)sampler;
    }
    NSLog(@"MGL SAMPLER ERROR: Metal-cpp sampler creation failed: %s",
          error[0] ? error : "unknown");
    return nil;
}

- (MGLMetalTextureRef)fallbackSampledTexture
{
    if (_resourceFallback.fallbackSampledTexture || !kMGLEnableSampledTextureFallback) {
        return _resourceFallback.fallbackSampledTexture;
    }

    MTLTextureDescriptor *desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:1
                                                          height:1
                                                       mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;

    _resourceFallback.fallbackSampledTexture =
        mglTextureCreateTexture(_device, desc);
    if (_resourceFallback.fallbackSampledTexture) {
        uint32_t pixel = 0xff000000u;
        mglTextureReplaceRegion(
            _resourceFallback.fallbackSampledTexture,
            MTLRegionMake2D(0, 0, 1, 1), 0, 0, &pixel,
            sizeof(pixel), 0, NO);
        NSLog(@"MGL INFO: Created 1x1 fallback sampled texture for missing shader resources");
    } else {
        NSLog(@"MGL ERROR: Failed to create fallback sampled texture");
    }

    return _resourceFallback.fallbackSampledTexture;
}

- (MGLMetalTextureRef)fallbackCubeSampledTexture
{
    if (_resourceFallback.fallbackCubeSampledTexture || !kMGLEnableSampledTextureFallback) {
        return _resourceFallback.fallbackCubeSampledTexture;
    }

    MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
    desc.textureType = MTLTextureTypeCube;
    desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
    desc.width = 1;
    desc.height = 1;
    desc.depth = 1;
    desc.arrayLength = 1;
    desc.mipmapLevelCount = 1;
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;

    _resourceFallback.fallbackCubeSampledTexture =
        mglTextureCreateTexture(_device, desc);
    if (_resourceFallback.fallbackCubeSampledTexture) {
        uint32_t pixel = 0xff000000u;
        for (NSUInteger face = 0; face < 6; face++) {
            mglTextureReplaceRegion(
                _resourceFallback.fallbackCubeSampledTexture,
                MTLRegionMake2D(0, 0, 1, 1), 0, face, &pixel,
                sizeof(pixel), sizeof(pixel), YES);
        }
        NSLog(@"MGL INFO: Created 1x1 fallback cube sampled texture for missing shader resources");
    } else {
        NSLog(@"MGL ERROR: Failed to create fallback cube sampled texture");
    }

    return _resourceFallback.fallbackCubeSampledTexture;
}

- (MGLMetalTextureRef)fallbackTextureBufferSampledTexture
{
    if (_resourceFallback.fallbackSintTextureBuffer || !kMGLEnableSampledTextureFallback) {
        return _resourceFallback.fallbackSintTextureBuffer;
    }

    static const NSUInteger kFallbackTexelCount = 64;
    static const NSUInteger kFallbackBytesPerTexel = 4;

    if (!_resourceFallback.fallbackTextureBufferStorage) {
        _resourceFallback.fallbackTextureBufferStorage = mglTextureCreateBuffer(
            _device, kFallbackTexelCount * kFallbackBytesPerTexel,
            MTLResourceStorageModeShared);
        if (_resourceFallback.fallbackTextureBufferStorage && _resourceFallback.fallbackTextureBufferStorage.contents) {
            memset(_resourceFallback.fallbackTextureBufferStorage.contents, 0, kFallbackTexelCount * kFallbackBytesPerTexel);
        }
    }

    if (!_resourceFallback.fallbackTextureBufferStorage) {
        NSLog(@"MGL ERROR: Failed to create fallback texture-buffer backing storage");
        return nil;
    }

    MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
    desc.textureType = MTLTextureTypeTextureBuffer;
    desc.pixelFormat = MTLPixelFormatRGBA8Sint;
    desc.width = kFallbackTexelCount;
    desc.height = 1;
    desc.depth = 1;
    desc.mipmapLevelCount = 1;
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;

    @try {
        _resourceFallback.fallbackSintTextureBuffer =
            mglTextureCreateBufferTexture(
                _resourceFallback.fallbackTextureBufferStorage, desc, 0,
                kFallbackTexelCount * kFallbackBytesPerTexel);
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Failed to create fallback texture-buffer texture: %@", exception);
        _resourceFallback.fallbackSintTextureBuffer = nil;
    }

    if (_resourceFallback.fallbackSintTextureBuffer) {
        NSLog(@"MGL INFO: Created fallback signed integer texture buffer for missing/invalid texel-buffer resources");
    }

    return _resourceFallback.fallbackSintTextureBuffer;
}

- (MGLMetalTextureRef)fallbackSampledTextureForExpectedType:(MTLTextureType)expectedType
                                               dataKind:(MGLTextureDataKind)dataKind
{
    if (!kMGLEnableSampledTextureFallback) {
        return nil;
    }

    MTLTextureType textureType = expectedType ? expectedType : MTLTextureType2D;
    if (textureType == MTLTextureTypeTextureBuffer) {
        return [self fallbackTextureBufferSampledTexture];
    }

    MTLPixelFormat pixelFormat = MTLPixelFormatRGBA8Unorm;
    if (dataKind == MGLTextureDataKindUint) {
        pixelFormat = MTLPixelFormatRGBA8Uint;
    } else if (dataKind == MGLTextureDataKindSint) {
        pixelFormat = MTLPixelFormatRGBA8Sint;
    } else if (dataKind == MGLTextureDataKindDepth) {
        pixelFormat = MTLPixelFormatDepth32Float;
    }

    if (!_resourceFallback.fallbackSampledTextureCache) {
        _resourceFallback.fallbackSampledTextureCache = [[NSMutableDictionary alloc] initWithCapacity:8];
    }

    NSUInteger keyValue = (((NSUInteger)textureType) << 8u) | ((NSUInteger)dataKind);
    NSNumber *key = @(keyValue);
    MGLMetalTextureRef cached = _resourceFallback.fallbackSampledTextureCache[key];
    if (cached) {
        return cached;
    }

    MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
    desc.textureType = textureType;
    desc.pixelFormat = pixelFormat;
    desc.width = 1;
    desc.height = 1;
    desc.depth = 1;
    desc.arrayLength = (textureType == MTLTextureTypeCube ||
                        textureType == MTLTextureTypeCubeArray ||
                        textureType == MTLTextureType2DArray ||
                        textureType == MTLTextureType1DArray ||
                        textureType == MTLTextureType2DMultisampleArray) ? 1 : 1;
    if (textureType == MTLTextureType2DMultisample ||
        textureType == MTLTextureType2DMultisampleArray) {
        desc.sampleCount = 2u;
    }
    desc.mipmapLevelCount = 1;
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;

    MGLMetalTextureRef texture = mglTextureCreateTexture(_device, desc);
    if (!texture) {
        NSLog(@"MGL ERROR: Failed to create %@ fallback sampled texture type=%lu format=%lu",
              [NSString stringWithUTF8String:mglTextureDataKindName(dataKind)],
              (unsigned long)textureType,
              (unsigned long)pixelFormat);
        return nil;
    }

    uint32_t pixel = dataKind == MGLTextureDataKindDepth ? 0u : 0xff000000u;
    MTLRegion region = textureType == MTLTextureType1D ||
                       textureType == MTLTextureType1DArray
        ? MTLRegionMake1D(0, 1)
        : MTLRegionMake2D(0, 0, 1, 1);
    if (textureType == MTLTextureTypeCube || textureType == MTLTextureTypeCubeArray) {
        NSUInteger sliceCount = (textureType == MTLTextureTypeCube) ? 6u : 6u;
        for (NSUInteger slice = 0; slice < sliceCount; slice++) {
            mglTextureReplaceRegion(
                texture, MTLRegionMake2D(0, 0, 1, 1), 0, slice,
                &pixel, sizeof(pixel), sizeof(pixel), YES);
        }
    } else if (textureType == MTLTextureType1DArray ||
               textureType == MTLTextureType2DArray) {
        mglTextureReplaceRegion(
            texture, region, 0, 0, &pixel, sizeof(pixel),
            sizeof(pixel), YES);
    } else {
        mglTextureReplaceRegion(
            texture, region, 0, 0, &pixel, sizeof(pixel), 0, NO);
    }

    _resourceFallback.fallbackSampledTextureCache[key] = texture;
    [self mglCapAuxCache:_resourceFallback.fallbackSampledTextureCache limit:32];
    NSLog(@"MGL INFO: Created %@ fallback sampled texture type=%lu format=%lu",
          [NSString stringWithUTF8String:mglTextureDataKindName(dataKind)],
          (unsigned long)textureType,
          (unsigned long)pixelFormat);

    return texture;
}


- (MGLMetalTextureRef)fallbackSampledTextureForExpectedType:(MTLTextureType)expectedType
{
    if (expectedType == MTLTextureTypeCube) {
        return [self fallbackCubeSampledTexture];
    }
    if (expectedType == MTLTextureTypeTextureBuffer) {
        return [self fallbackTextureBufferSampledTexture];
    }

    return [self fallbackSampledTexture];
}

- (int)textureIndexForExpectedMetalType:(MTLTextureType)expectedType
{
    return (int)mglRenderCppTextureIndexForMetalType((uint32_t)expectedType);
}

- (GLuint)textureUnitForSampledResource:(MGLShaderResource *)sampledResource
                                program:(Program *)program
                           metalBinding:(GLuint)metalBinding
                                  stage:(int)stage
{
    if (!program) {
        GLuint candidate = sampledResource &&
                           sampledResource->sampler_unit >= 0 &&
                           sampledResource->sampler_unit < TEXTURE_UNITS
            ? (GLuint)sampledResource->sampler_unit
            : metalBinding;
        return candidate;
    }

    const char *sampledName = NULL;
    if (!sampledResource && metalBinding < TEXTURE_UNITS) {
        sampledResource = mglFindSamplerResourceForMetalBinding(program, stage, metalBinding);
    }
    if (sampledResource) {
        sampledName = sampledResource->name;
    }

    /*
     * Minecraft usually assigns sampler texture units from the RenderPipeline
     * sampler list, not from numeric suffixes like Sampler2. For example, chunk
     * rendering declares Sampler0 and Sampler2, so Sampler2 can be uploaded
     * through glUniform1i(..., 1). Keep sampler units on the exact reflected
     * resource instead of only the Metal binding: vertex and fragment resources
     * commonly share binding numbers, and binding-level state can make entity,
     * hand, and text textures bleed into each other.
     */
    if (sampledResource &&
        sampledResource->sampler_unit_explicit &&
        sampledResource->sampler_unit >= 0 &&
        sampledResource->sampler_unit < TEXTURE_UNITS) {
        return (GLuint)sampledResource->sampler_unit;
    }

    if (metalBinding >= TEXTURE_UNITS) {
        return metalBinding;
    }

    bool stageExplicit = (stage >= 0 && stage < _MAX_SHADER_TYPES)
        ? (program->sampler_units_explicit_by_stage[stage][metalBinding] == GL_TRUE)
        : false;
    bool globalExplicit = (program->sampler_units_explicit[metalBinding] == GL_TRUE);

    GLint unit = (stage >= 0 && stage < _MAX_SHADER_TYPES)
        ? program->sampler_units_by_stage[stage][metalBinding]
        : program->sampler_units[metalBinding];

    if (stageExplicit && unit >= 0 && unit < TEXTURE_UNITS) {
        return (GLuint)unit;
    }

    unit = program->sampler_units[metalBinding];
    if (globalExplicit && unit >= 0 && unit < TEXTURE_UNITS) {
        return (GLuint)unit;
    }

    GLint defaultUnit = (stage >= 0 && stage < _MAX_SHADER_TYPES)
        ? program->sampler_units_by_stage[stage][metalBinding]
        : program->sampler_units[metalBinding];
    if (defaultUnit < 0 || defaultUnit >= TEXTURE_UNITS) {
        defaultUnit = program->sampler_units[metalBinding];
    }

    if (sampledResource &&
        !sampledResource->sampler_unit_explicit &&
        sampledResource->sampler_unit >= 0 &&
        sampledResource->sampler_unit < TEXTURE_UNITS) {
        return (GLuint)sampledResource->sampler_unit;
    }

    if (defaultUnit >= 0 && defaultUnit < TEXTURE_UNITS) {
        return (GLuint)defaultUnit;
    }

    /*
     * OpenGL's valid default is unit 0, and explicit glUniform1i uploads above
     * are authoritative. No name-based fallback is applied.
     */
    return 0u;
}

- (GLuint)textureUnitForSampledResource:(MGLShaderResource *)sampledResource metalBinding:(GLuint)metalBinding stage:(int)stage
{
    Program *program = mglResolveProgramForStageFromState(ctx, stage);
    return [self textureUnitForSampledResource:sampledResource
                                      program:program
                                 metalBinding:metalBinding
                                        stage:stage];
}

- (GLuint)textureUnitForSampledBinding:(GLuint)metalBinding stage:(int)stage
{
    return [self textureUnitForSampledResource:NULL metalBinding:metalBinding stage:stage];
}

- (Texture *)textureForSampledResource:(MGLShaderResource *)sampledResource
                          metalBinding:(GLuint)metalBinding
                                  stage:(int)stage
                           expectedType:(MTLTextureType)expectedType
                          textureUnit:(GLuint)textureUnit
{
    if (!ctx || metalBinding >= TEXTURE_UNITS) {
        return NULL;
    }

    if (textureUnit >= TEXTURE_UNITS) {
        return NULL;
    }

    if (expectedType == 0) {
        return MGL_STATE(ctx)->active_textures[textureUnit];
    }

    int textureIndex = [self textureIndexForExpectedMetalType:expectedType];
    if (textureIndex >= 0 && textureIndex < _MAX_TEXTURE_TYPES) {
        Texture *typedTexture = MGL_STATE(ctx)->texture_units[textureUnit].textures[textureIndex];
        /* The AIR backend lowers sampler1D to texture2d, so expectedType is
         * MTLTextureType2D even for GL_TEXTURE_1D bindings. If the _TEXTURE_2D
         * slot only contains an auto-created default texture (name ==
         * TEX_OBJ_RES_NAME) while the unit's active texture is a real
         * GL_TEXTURE_1D, prefer the 1D texture. Otherwise the default 2D
         * texture leaks across test cases and masks the real 1D binding. */
        if (typedTexture && typedTexture->name == TEX_OBJ_RES_NAME) {
            Texture *activeTexture = MGL_STATE(ctx)->active_textures[textureUnit];
            if (activeTexture && activeTexture->name != TEX_OBJ_RES_NAME) {
                typedTexture = NULL;
            }
        }
        if (typedTexture) {
            return typedTexture;
        }

        if (expectedType == MTLTextureType2D) {
            Texture *activeTexture = MGL_STATE(ctx)->active_textures[textureUnit];
            if (activeTexture &&
                activeTexture->target == GL_TEXTURE_1D) {
                return activeTexture;
            }
        }

        // Texel-buffer resources must not silently fall back to GL_TEXTURE_2D.
        // Minecraft's CloudFaces is declared with buffer image_dim, and the lowerer
        // lowers it to a 1-row texture2d<int> in MSL. If no GL_TEXTURE_BUFFER
        // is bound, using the active 2D atlas here feeds float/RGBA data into a
        // signed integer vertex resource and corrupts the whole frame.
        if (expectedType == MTLTextureTypeTextureBuffer) {
            static uint64_t s_missingTextureBufferBindingLogs = 0;
            uint64_t hit = ++s_missingTextureBufferBindingLogs;
            if (hit <= 32ull || (hit % 512ull) == 0ull) {
                Texture *activeTexture = MGL_STATE(ctx)->active_textures[textureUnit];
                NSLog(@"MGL TEXBUFFER BIND MISSING binding=%u unit=%u activeTex=%u activeTarget=0x%x hit=%llu",
                      (unsigned)metalBinding,
                      (unsigned)textureUnit,
                      activeTexture ? (unsigned)activeTexture->name : 0u,
                      activeTexture ? (unsigned)activeTexture->target : 0u,
                      (unsigned long long)hit);
            }
            return NULL;
        }

        /*
         * OpenGL texture units keep one binding per texture target. A sampler2D
         * samples the GL_TEXTURE_2D slot for its unit, even if a cubemap or texel
         * buffer was bound more recently on that same unit. Falling back to the
         * unit's "active" texture here lets sky cubemaps and buffer textures bleed
         * into item/entity shaders when Minecraft switches pipelines.
         */
        static uint64_t s_missingTypedTextureBindingLogs = 0;
        uint64_t hit = ++s_missingTypedTextureBindingLogs;
        if (hit <= 64ull || (hit % 512ull) == 0ull) {
            Texture *activeTexture = MGL_STATE(ctx)->active_textures[textureUnit];
            NSLog(@"MGL TEX TYPED BIND MISSING binding=%u stage=%s unit=%u expectedType=%lu expectedIndex=%d activeTex=%u activeTarget=0x%x hit=%llu",
                  (unsigned)metalBinding,
                  mglShaderStageName(stage),
                  (unsigned)textureUnit,
                  (unsigned long)expectedType,
                  textureIndex,
                  activeTexture ? (unsigned)activeTexture->name : 0u,
                  activeTexture ? (unsigned)activeTexture->target : 0u,
                  (unsigned long long)hit);
        }
        return NULL;
    }

    return MGL_STATE(ctx)->active_textures[textureUnit];
}

- (Texture *)textureForSampledResource:(MGLShaderResource *)sampledResource
                          metalBinding:(GLuint)metalBinding
                                  stage:(int)stage
                           expectedType:(MTLTextureType)expectedType
{
    if (!ctx || metalBinding >= TEXTURE_UNITS) {
        return NULL;
    }
    GLuint textureUnit = [self textureUnitForSampledResource:sampledResource
                                                metalBinding:metalBinding
                                                       stage:stage];
    return [self textureForSampledResource:sampledResource
                              metalBinding:metalBinding
                                      stage:stage
                               expectedType:expectedType
                              textureUnit:textureUnit];
}

- (Texture *)textureForSampledBinding:(GLuint)metalBinding stage:(int)stage expectedType:(MTLTextureType)expectedType
{
    return [self textureForSampledResource:NULL
                              metalBinding:metalBinding
                                      stage:stage
                               expectedType:expectedType];
}

- (MGLMetalSamplerStateRef)fallbackSamplerState
{
    if (_resourceFallback.fallbackSamplerState) {
        return _resourceFallback.fallbackSamplerState;
    }

    MTLSamplerDescriptor *desc = [MTLSamplerDescriptor new];
    desc.minFilter = MTLSamplerMinMagFilterNearest;
    desc.magFilter = MTLSamplerMinMagFilterNearest;
    desc.mipFilter = MTLSamplerMipFilterNotMipmapped;
    desc.sAddressMode = MTLSamplerAddressModeClampToEdge;
    desc.tAddressMode = MTLSamplerAddressModeClampToEdge;
    desc.rAddressMode = MTLSamplerAddressModeClampToEdge;

    _resourceFallback.fallbackSamplerState =
        mglTextureCreateSampler(_device, desc);
    if (!_resourceFallback.fallbackSamplerState) {
        NSLog(@"MGL ERROR: Failed to create fallback sampler state");
    }

    return _resourceFallback.fallbackSamplerState;
}

- (void)traceSampledTextureReadback:(MGLMetalTextureRef)texture
                              glTex:(Texture *)glTex
                              level:(TextureLevel *)level0
                            program:(GLuint)program
                            binding:(GLuint)binding
                              stage:(NSString *)stage
                             reason:(NSString *)reason
                                hit:(uint64_t)hit
{
    if (!texture || !_device || !_commandQueue) {
        return;
    }

    MTLPixelFormat fmt = texture.pixelFormat;
    BOOL fourByteColor =
        fmt == MTLPixelFormatRGBA8Unorm ||
        fmt == MTLPixelFormatRGBA8Unorm_sRGB ||
        fmt == MTLPixelFormatBGRA8Unorm ||
        fmt == MTLPixelFormatBGRA8Unorm_sRGB;
    if (!fourByteColor) {
        mglTraceLogNSString(@"MGL TRACE sampled.readback skip program=%u binding=%u glTex=%u reason=%@ fmt=%lu type=%lu size=%lux%lu hit=%llu",
              (unsigned)program,
              (unsigned)binding,
              glTex ? (unsigned)glTex->name : 0u,
              reason,
              (unsigned long)fmt,
              (unsigned long)texture.textureType,
              (unsigned long)texture.width,
              (unsigned long)texture.height,
              (unsigned long long)hit);
        return;
    }

    NSUInteger texWidth = (NSUInteger)texture.width;
    NSUInteger texHeight = (NSUInteger)texture.height;
    if (texWidth == 0 || texHeight == 0) {
        return;
    }

    NSUInteger sampleWidth = MIN(texWidth, 8u);
    NSUInteger sampleHeight = MIN(texHeight, 8u);
    NSUInteger bytesPerPixel = 4u;
    NSUInteger bytesPerRow = sampleWidth * bytesPerPixel;
    NSUInteger byteCount = bytesPerRow * sampleHeight;
    if (byteCount == 0) {
        return;
    }

    MGLMetalBufferRef readback = mglTextureCreateBuffer(
        _device, byteCount, MTLResourceStorageModeShared);
    MGLMetalCommandBufferRef cb = mglTextureCreateCommandBuffer(_commandQueue);
    MGLMetalBlitCommandEncoderRef blit = mglTextureCreateBlitEncoder(cb);
    if (!readback || !cb || !blit) {
        mglTraceLogNSString(@"MGL TRACE sampled.readback setup-fail program=%u binding=%u glTex=%u reason=%@ readback=%p cb=%p blit=%p hit=%llu",
              (unsigned)program,
              (unsigned)binding,
              glTex ? (unsigned)glTex->name : 0u,
              reason,
              readback,
              cb,
              blit,
              (unsigned long long)hit);
        return;
    }

    mglTextureCopyTextureToBuffer(
        blit, texture, 0, 0, MTLOriginMake(0, 0, 0),
        MTLSizeMake(sampleWidth, sampleHeight, 1), readback, 0,
        bytesPerRow, byteCount);
    mglTextureEndBlitEncoder(blit);
    mglTextureCommitCommandBuffer(cb);
    mglTextureWaitCommandBuffer(cb);

    const uint8_t *p = (const uint8_t *)readback.contents;
    uint64_t byteSum = 0;
    NSUInteger nonZeroBytes = 0;
    uint32_t firstPixel = 0;
    uint32_t pixelXor = 0;
    uint32_t minPixel = UINT32_MAX;
    uint32_t maxPixel = 0;
    NSUInteger pixelCount = byteCount / sizeof(uint32_t);

    if (p) {
        for (NSUInteger i = 0; i < byteCount; i++) {
            byteSum += (uint64_t)p[i];
            if (p[i] != 0) {
                nonZeroBytes++;
            }
        }
        if (byteCount >= sizeof(firstPixel)) {
            memcpy(&firstPixel, p, sizeof(firstPixel));
        }
        for (NSUInteger i = 0; i < pixelCount; i++) {
            uint32_t pixel = 0;
            memcpy(&pixel, p + (i * sizeof(pixel)), sizeof(pixel));
            pixelXor ^= pixel;
            if (pixel < minPixel) {
                minPixel = pixel;
            }
            if (pixel > maxPixel) {
                maxPixel = pixel;
            }
        }
    }

    MGLRenderCppCommandBufferState sampledState =
        mglRenderCommandBufferState(cb);
    NSString *sampledError = sampledState.has_error
        ? [NSString stringWithFormat:@"%s (domain=%s code=%lld)",
             sampledState.error_description,
             sampledState.error_domain,
             (long long)sampledState.error_code]
        : nil;
    mglTraceLogNSString(@"MGL TRACE sampled.readback stage=%@ program=%u binding=%u glTex=%u reason=%@ hit=%llu "
          "mtl=%p fmt=%lu type=%lu size=%lux%lu sample=%lux%lu status=%s error=%@ "
          "nonZero=%lu/%lu sum=%llu first=0x%08x min=0x%08x max=0x%08x xor=0x%08x "
          "level(init ever=%u full=%u zero=%u source=%u upload=%lu src=%p hash=0x%016llx)",
          stage,
          (unsigned)program,
          (unsigned)binding,
          glTex ? (unsigned)glTex->name : 0u,
          reason,
          (unsigned long long)hit,
          texture,
          (unsigned long)fmt,
          (unsigned long)texture.textureType,
          (unsigned long)texWidth,
          (unsigned long)texHeight,
          (unsigned long)sampleWidth,
          (unsigned long)sampleHeight,
          mglCommandBufferStatusName(
              (MTLCommandBufferStatus)sampledState.status),
          sampledError,
          (unsigned long)nonZeroBytes,
          (unsigned long)byteCount,
          (unsigned long long)byteSum,
          firstPixel,
          minPixel == UINT32_MAX ? 0u : minPixel,
          maxPixel,
          pixelXor,
          level0 ? (unsigned)level0->ever_written : 0u,
          level0 ? (unsigned)level0->has_initialized_data : 0u,
          level0 ? (unsigned)level0->suspicious_zero_upload : 0u,
          level0 ? (unsigned)level0->last_init_source : 0u,
          (unsigned long)(level0 ? level0->last_upload_size : 0u),
          level0 ? (void *)level0->last_src_ptr : NULL,
          (unsigned long long)(level0 ? level0->last_src_hash : 0ull));
}
@end
