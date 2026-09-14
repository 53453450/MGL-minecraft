/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+Texture.m
// Texture upload/download Metal path methods extracted from MGLRenderer.m

#import "MGLRenderer_Private.h"
#include "mgl_blit_color_state.h" /* readback helper entries (log 141) */
#include "mgl_texture_readback_clear.h"
#include "mgl_gpu_recovery.h"
#include "mgl_pixel_format.h"
#include "mgl_texture_binding_resolve.h"
#import "MGLRenderer+Texture_Private.h"
#import "mgl_texture_readback_ops.h" /* the readback family is C now (log 181) */
#import "mgl_texture_create_ops.h" /* completeness / packed-DS upload / texel buffer (log 182) */
#import "mgl_texture_upload_ops.h" /* slice upload + dedicated CB copy (log 183) */
#import "mgl_texture_readback_ops.h" /* readPixels family is C (log 185) */
#include "mgl_env_flag.h"
#include "mgl_render.h"
#include "mgl_renderer_ports.h"  /* mglRendererProcessBuffer */
#include "mgl_blit_sampled_copy.h"  /* sampled RT copy refresh */
#include "mgl_blit_drivers.h"   /* mglBlitCopyImageSubData (log 145) */
#include "mgl_region_value.h"   // canonical region/origin/size constructors (O4 dedup sink)

enum {
    MGL_TEXTURE_RESOURCE_STORAGE_SHARED = 0u,
    MGL_TEXTURE_STORAGE_PRIVATE = 2u,
    MGL_TEXTURE_CPU_CACHE_DEFAULT = 0u,
    MGL_TEXTURE_CPU_CACHE_WRITE_COMBINED = 1u,
    MGL_TEXTURE_USAGE_SHADER_READ = 1u,
    MGL_TEXTURE_USAGE_SHADER_WRITE = 2u,
    /* Matches MTLTextureUsageShaderAtomic (macOS 14+ / Metal 3.1). */
    MGL_TEXTURE_USAGE_SHADER_ATOMIC = 0x20u,
    MGL_TEXTURE_USAGE_RENDER_TARGET = 4u,
    MGL_TEXTURE_USAGE_PIXEL_FORMAT_VIEW = 16u,
};

static MGLRegionValue mglRendererCompatRegion(int32_t x, int32_t y,
                                         int32_t width, int32_t height)
{
    return (MGLRegionValue){
        .origin = {(int64_t)x, (int64_t)y, 0},
        .size = {(uint64_t)width, (uint64_t)height, 1u},
    };
}

void mglRendererReadDrawable(GLMContext glm_ctx, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        mglTextureReadDrawable((__bridge void *)renderer, glm_ctx,
                               pixel_bytes, bytes_per_row, bytes_per_image,
                               mglRendererCompatRegion(x, y, width, height));
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererReadIntegerPixels(GLMContext glm_ctx, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height,
    uint32_t format, uint32_t type)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        mglTextureReadIntegerPixels((__bridge void *)renderer, glm_ctx,
                                    pixel_bytes, bytes_per_row,
                                    bytes_per_image,
                                    mglRendererCompatRegion(x, y, width, height),
                                    format, type);
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererReadDepthPixels(GLMContext glm_ctx, void *pixel_bytes,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        mglTextureReadDepthPixels((__bridge void *)renderer, glm_ctx,
                                  pixel_bytes, bytes_per_row, bytes_per_image,
                                  mglRendererCompatRegion(x, y, width, height));
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererGetTexImage(GLMContext glm_ctx, Texture *texture,
    void *pixel_bytes, uint32_t bytes_per_row, uint32_t bytes_per_image,
    int32_t x, int32_t y, int32_t width, int32_t height,
    uint32_t format, uint32_t type, uint32_t level, uint32_t slice)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        [renderer mtlGetTexImage:glm_ctx tex:texture pixelBytes:pixel_bytes
                     bytesPerRow:bytes_per_row bytesPerImage:bytes_per_image
                      fromRegion:mglRendererCompatRegion(x, y, width, height)
                          format:format type:type mipmapLevel:level slice:slice];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererGenerateMipmaps(GLMContext glm_ctx, Texture *texture)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        [renderer mtlGenerateMipmaps:glm_ctx forTexture:texture];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererSyncTextureBufferFromImage(GLMContext glm_ctx, Texture *texture)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx && texture) {
        [renderer syncTextureBufferFromImage:glm_ctx tex:texture];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererPrepareImageUnitSlice(GLMContext glm_ctx, uint32_t unit)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        [renderer prepareImageUnitSlice:glm_ctx unit:unit];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererFlushImageUnitSlice(GLMContext glm_ctx, uint32_t unit)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        [renderer flushImageUnitSlice:glm_ctx unit:unit];
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererTexSubImage(GLMContext glm_ctx, Texture *texture, Buffer *buffer,
    size_t source_offset, size_t source_pitch, size_t source_image_size,
    size_t source_size, uint32_t slice, uint32_t level,
    size_t width, size_t height, size_t depth,
    size_t x_offset, size_t y_offset, size_t z_offset)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        [renderer mtlTexSubImage:glm_ctx tex:texture buf:buffer
                      src_offset:source_offset src_pitch:source_pitch
                  src_image_size:source_image_size src_size:source_size
                           slice:slice level:level width:width height:height
                           depth:depth xoffset:x_offset yoffset:y_offset
                         zoffset:z_offset];
    }
    mglRendererBackendEnd(&_backend_lease);
}

bool mglRendererTexSubImageBytes(GLMContext glm_ctx, Texture *texture,
    const void *bytes, size_t bytes_size,
    size_t source_offset, size_t source_pitch, size_t source_image_size,
    uint32_t slice, uint32_t level,
    size_t width, size_t height, size_t depth,
    size_t x_offset, size_t y_offset, size_t z_offset)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return false;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    bool result = false;
    if (renderer && glm_ctx) {
        result = [renderer mtlTexSubImageBytes:glm_ctx tex:texture
                                    bytes:bytes bytesSize:bytes_size
                               src_offset:source_offset src_pitch:source_pitch
                           src_image_size:source_image_size
                                    slice:slice level:level
                                    width:width height:height depth:depth
                                  xoffset:x_offset yoffset:y_offset
                                  zoffset:z_offset];
    }
    mglRendererBackendEnd(&_backend_lease);
    return result;
}

void mglRendererCopyTexSubImage(GLMContext glm_ctx, Texture *texture,
    uint32_t slice, int32_t level, int32_t x_offset, int32_t y_offset,
    int32_t x, int32_t y, int32_t width, int32_t height)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        mglBlitCopyTexSubImage((__bridge void *)renderer, glm_ctx, texture,
                               slice, level, x_offset, y_offset, x, y, width,
                               height);
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererCopyImageSubData(GLMContext glm_ctx, Texture *source_texture,
    int32_t source_level, int32_t source_x, int32_t source_y, int32_t source_z,
    Texture *destination_texture, int32_t destination_level,
    int32_t destination_x, int32_t destination_y, int32_t destination_z,
    int32_t width, int32_t height, int32_t depth)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        mglBlitCopyImageSubData(
            (__bridge void *)renderer, glm_ctx, source_texture, source_level,
            source_x, source_y, source_z, destination_texture,
            destination_level, destination_x, destination_y, destination_z,
            width, height, depth);
    }
    mglRendererBackendEnd(&_backend_lease);
}

static id mglTextureCreateBuffer(id device,
                                            NSUInteger length,
                                            uint64_t options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCreateBuffer(length, options, NULL, &buffer) == 0 &&
        buffer) {
        return (__bridge_transfer id)buffer;
    }
    return nil;
}

static id mglTextureCreateBufferWithBytes(
    id device,
    const void *bytes,
    NSUInteger length,
    uint64_t options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCreateBufferWithBytes(bytes, length, options, NULL,
                                          &buffer) == 0 && buffer) {
        return (__bridge_transfer id)buffer;
    }
    return nil;
}

static id mglTextureCreateTexture(
    id device,
    const MGLRenderTextureDescriptorState *descriptor)
{
    (void)device;
    void *texture = NULL;
    if (mglRenderCreateTextureFromState(
            descriptor, NULL, &texture) == 0 &&
        texture) {
        return (__bridge_transfer id)texture;
    }
    return nil;
}

static id mglTextureCreateBufferTexture(
    id buffer,
    const MGLRenderTextureDescriptorState *descriptor,
    NSUInteger offset,
    NSUInteger bytesPerRow)
{
    void *texture = NULL;
    if (mglRenderCreateBufferTextureFromState(
            (__bridge void *)buffer, descriptor,
            offset, bytesPerRow,
            &texture) == 0 && texture) {
        return (__bridge_transfer id)texture;
    }
    return nil;
}

static void mglTextureReplaceRegion(id texture,
                                    MGLRegionValue region,
                                    NSUInteger level,
                                    NSUInteger slice,
                                    const void *bytes,
                                    NSUInteger bytesPerRow,
                                    NSUInteger bytesPerImage,
                                    BOOL useSlice)
{
    if (mglRenderTextureReplaceRegion(
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

static void mglTextureGetBytes(id texture,
                               void *bytes,
                               NSUInteger bytesPerRow,
                               NSUInteger bytesPerImage,
                               MGLRegionValue region,
                               NSUInteger level,
                               NSUInteger slice,
                               BOOL useSlice)
{
    if (mglRenderTextureGetBytes(
            (__bridge void *)texture, bytes, bytesPerRow, bytesPerImage,
            region.origin.x, region.origin.y, region.origin.z,
            region.size.width, region.size.height, region.size.depth,
            level, slice, useSlice ? 1 : 0) != 0) {
        [NSException raise:@"MGLTextureGetBytesError"
                    format:@"C++ texture getBytes failed (level=%lu slice=%lu)",
                           (unsigned long)level, (unsigned long)slice];
    }
}


static id mglTextureCreateCommandBuffer(
    id queue)
{
    if (!queue) return nil;
    void *commandBuffer = NULL;
    if (mglRenderCreateCommandBuffer((__bridge void *)queue,
                                         &commandBuffer) == 0 &&
        commandBuffer) {
        return (__bridge id)commandBuffer;
    }
    return nil;
}

static id mglTextureCreateBlitEncoder(
    id commandBuffer)
{
    if (!commandBuffer) return nil;
    void *encoder = NULL;
    if (mglRenderCreateBlitEncoder((__bridge void *)commandBuffer,
                                       &encoder) == 0 && encoder) {
        return (__bridge id)encoder;
    }
    return nil;
}

/* Owner-first adapter for work that is encoded on the renderer's current
 * command buffer. Dedicated command buffers continue to use the raw helper
 * above because they are not owned by MGLRenderPassManager. */
static id mglTextureCreateCurrentBlitEncoder(
    void *commandBufferOwner)
{
    return (__bridge id)mglRenderCreateBlitEncoderBorrowed(
        commandBufferOwner);
}

static void mglTextureEndBlitEncoder(id encoder)
{
    if (!encoder) return;
    (void)mglRenderEndBlitEncoder((__bridge void *)encoder);
}

static void mglTextureCommitCommandBuffer(id commandBuffer)
{
    if (!commandBuffer) return;
    if (mglRenderCommitCommandBuffer(
            (__bridge void *)commandBuffer) != 0) {
        NSLog(@"MGL ERROR: Metal-cpp texture command-buffer commit failed");
    }
}

static void mglTextureWaitCommandBuffer(id commandBuffer)
{
    if (!commandBuffer) return;
    if (mglRenderWaitCommandBuffer(
            (__bridge void *)commandBuffer) != 0) {
        NSLog(@"MGL ERROR: Metal-cpp texture command-buffer wait failed");
    }
}

static MGLRenderTextureInfo mglTextureInfo(id texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) (void)mglRenderGetTextureInfo((__bridge void *)texture, &info);
    return info;
}

/* AGX replaceRegion/copyFromBuffer require 256-byte row alignment for many
 * depth/stencil pixel formats even when the logical row is smaller. */
static const NSUInteger kMGLDepthStencilUploadRowAlignment = 256u;

static NSUInteger mglDepthStencilAlignedBytesPerRow(NSUInteger logicalBytesPerRow)
{
    if (logicalBytesPerRow == 0) {
        return 0;
    }
    return ((logicalBytesPerRow + kMGLDepthStencilUploadRowAlignment - 1u) /
            kMGLDepthStencilUploadRowAlignment) * kMGLDepthStencilUploadRowAlignment;
}

/* CPU shadow storage uses five bytes per texel for GL_DEPTH32F_STENCIL8
 * (float depth plus one stencil byte), while Metal's packed depth/stencil
 * upload layout uses an eight-byte texel with stencil at byte 4. */
static void *mglCreateDepthStencilMetalUpload(
    Texture *tex, uint32_t pixelFormat, const uint8_t *src,
    NSUInteger width, NSUInteger height, NSUInteger srcBytesPerRow,
    NSUInteger *outBytesPerRow, NSUInteger *outBytesPerImage)
{
    if (outBytesPerRow) *outBytesPerRow = 0;
    if (outBytesPerImage) *outBytesPerImage = 0;
    if (!tex || !src || width == 0 || height == 0 || srcBytesPerRow == 0 ||
        !mglRenderDepth32FStencil8NeedsUnpack(
            (uint32_t)tex->internalformat, (uint32_t)pixelFormat,
            (uint32_t)srcBytesPerRow, (uint32_t)width)) {
        return NULL;
    }
    NSUInteger logicalBytesPerRow = width * 8u;
    NSUInteger dstBytesPerRow = mglDepthStencilAlignedBytesPerRow(logicalBytesPerRow);
    if (dstBytesPerRow == 0) {
        return NULL;
    }
    NSUInteger dstBytesPerImage = dstBytesPerRow * height;
    uint8_t *dst = calloc(1u, dstBytesPerImage);
    if (!dst) return NULL;
    for (NSUInteger y = 0; y < height; ++y) {
        const uint8_t *srcRow = src + y * srcBytesPerRow;
        uint8_t *dstRow = dst + y * dstBytesPerRow;
        for (NSUInteger x = 0; x < width; ++x) {
            memcpy(dstRow + x * 8u, srcRow + x * 5u, 4u);
            dstRow[x * 8u + 4u] = srcRow[x * 5u + 4u];
        }
    }
    if (outBytesPerRow) *outBytesPerRow = dstBytesPerRow;
    if (outBytesPerImage) *outBytesPerImage = dstBytesPerImage;
    return dst;
}


static void *mglTextureBufferContents(id buffer)
{
    void *contents = NULL;
    uint64_t length = 0u;
    return buffer && mglRenderGetBufferContents((__bridge void *)buffer,
                                                   &contents, &length) == 0
        ? contents : NULL;
}

static void mglTextureCopyTextureToBuffer(
    id encoder,
    id source,
    NSUInteger sourceSlice,
    NSUInteger sourceLevel,
    MGLOriginValue sourceOrigin,
    MGLSizeValue sourceSize,
    id destination,
    NSUInteger destinationOffset,
    NSUInteger bytesPerRow,
    NSUInteger bytesPerImage)
{
    (void)mglRenderBlitCopyTextureToBuffer(
            (__bridge void *)encoder, (__bridge void *)source, sourceSlice,
            sourceLevel, sourceOrigin.x, sourceOrigin.y, sourceOrigin.z,
            sourceSize.width, sourceSize.height, sourceSize.depth,
            (__bridge void *)destination, destinationOffset, bytesPerRow,
            bytesPerImage);
}

@implementation MGLRenderer (Texture)

- (bool)uploadFullCPUTextureDataIntoTexture:(Texture *)tex
                                      metal:(id)texture
                                     reason:(const char *)reason
{
    if (!tex || !texture || !tex->faces[0].levels) {
        return false;
    }
    if (!mglRenderTextureTargetIs2D((uint32_t)tex->target) ||
        mglTextureInfo(texture).texture_type != MGLTextureType2D) {
        return false;
    }

    int numFaces = 1;
    GLuint levelCount = MIN((GLuint)mglTextureInfo(texture).mipmap_level_count,
                            tex->num_levels ? tex->num_levels : 1u);
    if (levelCount == 0u ||
        !mglTextureHasUploadableCPUData(tex, numFaces, levelCount)) {
        return false;
    }


    MGLRenderLevelUploadOp uploadOps[levelCount ? levelCount : 1u];
    uint32_t opCount = 0;
    uint32_t shortCount = 0;
    uint32_t badCount = 0;
    if (mglRenderBuildLevelUploadOps(
            tex->faces[0].levels, levelCount,
            (uint32_t)mglTextureInfo(texture).texture_type,
            (uint32_t)tex->internalformat,
            (uint32_t)mglTextureInfo(texture).pixel_format,
            uploadOps, levelCount,
            &opCount, &shortCount, &badCount) != 0) {
        return false;
    }

    bool uploadedAny = false;
    bool failedAny = (shortCount + badCount) > 0;
    for (uint32_t i = 0; i < opCount; i++) {
        MGLRenderLevelUploadOp *op = &uploadOps[i];
        if (op->kind == 1u) {
            static uint64_t s_shortBackingLogs = 0;
            uint64_t hit = ++s_shortBackingLogs;
            if (kMGLDiagnosticStateLogs &&
                (hit <= 32ull || (hit % 512ull) == 0ull)) {
                mglTraceLog("MGL TEXTURE CPU-REFRESH skip short backing tex=%u level=%u face=0 have=%llu need=%llu reason=%s hit=%llu",
                              (unsigned)tex->name,
                              (unsigned)op->level,
                              (unsigned long long)op->available_bytes,
                              (unsigned long long)op->needed_bytes,
                              reason ? reason : "(null)",
                              (unsigned long long)hit);
            }
            continue;
        }

        bool uploaded = mglTextureUploadSliceViaBlit(
            (__bridge void *)self, (__bridge void *)texture, tex->name, tex->target, op->data, (NSUInteger)op->bytes_per_row, (NSUInteger)op->bytes_per_image, (NSUInteger)op->width, (NSUInteger)op->height, (NSUInteger)op->copy_depth, op->level, 0);
        if (op->owns_data) {
            free((void *)op->data);
        }
        if (uploaded) {
            uploadedAny = true;
            /* CPU refreshed Metal mip while a Y-flip sampled copy may still
             * hold the previous RT contents for that level. */
            mglMarkGLSampledCopyLevelDirty(tex, op->level);
        } else {
            failedAny = true;
        }
    }

    static uint64_t s_refreshLogs = 0;
    uint64_t hit = ++s_refreshLogs;
    if (kMGLDiagnosticStateLogs &&
        (uploadedAny || hit <= 32ull || (hit % 512ull) == 0ull)) {
        mglTraceLog("MGL TEXTURE CPU-REFRESH tex=%u mtl=%p uploaded=%d failed=%d dirty=0x%x levels=%u reason=%s hit=%llu",
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
        mglRendererRecordGPUSuccess((__bridge void *)self);
        return true;
    }

    return false;
}


-(void) mtlGetTexImage:(GLMContext) glm_ctx tex: (Texture *)tex pixelBytes:(void *)pixelBytes bytesPerRow:(NSUInteger)bytesPerRow bytesPerImage:(NSUInteger)bytesPerImage fromRegion:(MGLRegionValue)region format:(GLenum)format type:(GLenum)type mipmapLevel:(NSUInteger)level slice:(NSUInteger)slice
{
    id texture = nil;

    ctx = glm_ctx;

    if (!tex) {
        NSLog(@"MGL ERROR: mtlGetTexImage called with NULL texture");
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    if (!pixelBytes) {
        NSLog(@"MGL WARNING: mtlGetTexImage called with NULL destination for texture %u", tex->name);
        return;
    }

    if (!tex->mtl_data && ![self bindMTLTexture:tex]) {
        NSLog(@"MGL ERROR: mtlGetTexImage failed to bind texture %u", tex->name);
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    texture = (__bridge id)(tex->mtl_data);
    if (!texture) {
        NSLog(@"MGL ERROR: mtlGetTexImage texture %u has no Metal texture", tex->name);
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    if (mglRenderTextureIsFramebufferOnly((__bridge void *)texture)) {
        NSLog(@"MGL ERROR: Cannot read from framebuffer only texture %u\n", tex->name);
        mglDispatchError(ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    if (![self synchronizeRenderPassForTextureReadback:texture reason:"mtlGetTexImage"]) {
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    /* Ensure any pending texture upload blit commands are committed before
     * reading back. Without this, getBytes may return stale/zero data because
     * the blit encoding the upload is still in the uncommitted command buffer. */
    [self endRenderEncoding];
    if (mglRenderCommandBufferOwnerHasCurrent(
            _renderPassManager->state->currentCommandBufferOwner) == 1) {
        id pendingCB =
            (__bridge id)mglPassManagerDetachCurrentCommandBufferForSubmission(_renderPassManager);
        @try {
            mglRendererCommitCommandBufferWithAGXRecovery((__bridge void *)self, (__bridge void *)pendingCB);
            mglTextureWaitCommandBuffer(pendingCB);
        } @catch (NSException *e) {
            NSLog(@"MGL WARNING: mtlGetTexImage pre-readback flush failed: %@", e.reason);
        }
        MGLRenderCommandBufferState pendingState = {0};
        (void)mglRenderGetCommandBufferState(
            (__bridge void *)pendingCB, &pendingState);
        if (pendingState.has_error) {
            NSLog(@"MGL WARNING: mtlGetTexImage pre-readback command buffer error: %s",
                  mglRenderCommandBufferErrorDescription(&pendingState));
        }
        [self newCommandBuffer];
    }

    MGLRegionValue readRegion = region;
    NSUInteger readSlice = slice;
    /* TextureType3D uses origin.z for depth planes; arrayLength is always 1.
     * Callers pass the depth index via `slice` (see mglGetTexImage's layer
     * loop). Remap so blit uses slice=0 and origin.z = depth plane. */
    if (mglTextureInfo(texture).texture_type == MGLTextureType3D) {
        readRegion.origin.z = slice;
        if (readRegion.size.depth < 1u) {
            readRegion.size.depth = 1u;
        }
        readSlice = 0u;
    }
    /* Single-sample pass-rendered RTs are stored top-row-first in Metal
     * (NDC y=-1 at high row addresses) and need a CPU Y-flip for GL's
     * bottom-up readPixels.  Multisample RTs already land in GL row order
     * after resolve — flipping them re-inverts DSA MSAA float/unorm
     * getTexImage (3a8cb5c). */
    BOOL flipRenderTargetRows =
        tex->is_render_target && tex->samples <= 1u;

    /* Integer texture readback path: when the source texture is an integer
     * format and the output format is GL_*_INTEGER, use the dedicated integer
     * readback function that handles packed types and component mapping. */

    MGLRenderIntegerReadbackClassify classify = {0};
    mglRenderIntegerReadbackClassify(
        (uint32_t)mglTextureInfo(texture).pixel_format, (uint32_t)format, (uint32_t)type,
        &classify);

    if (classify.source_is_integer_texture &&
        classify.output_is_integer_format) {
        /* Pass the original (non-Y-flipped) region. mglReadIntegerTextureAsRGBA32
         * does its own Y-flip on the blit source origin AND Y-flips the output
         * rows, so passing a pre-Y-flipped readRegion here would double-flip. */
        mglTextureReadIntegerAsRGBA32(
            (__bridge void *)self, (__bridge void *)texture, pixelBytes, bytesPerRow, bytesPerImage, readRegion, (NSUInteger)classify.output_components, (NSUInteger)classify.output_component_bytes, classify.component_map, type, level, readSlice, tex->is_render_target ? 1 : 0);
        return;
    }

    NSUInteger dstPixelBytes = (NSUInteger)sizeForFormatType(format, type);
    BOOL directR32FloatRead =
        mglRenderDirectR32FloatRead(
            (uint32_t)mglTextureInfo(texture).pixel_format, (uint32_t)format,
            (uint32_t)type) != 0;
    BOOL useBGRA8Conversion =
        (dstPixelBytes > 0u &&
         readRegion.size.depth == 1u &&
         !directR32FloatRead &&
         mglMetalReadbackFormatIsBGRA8Compatible(mglTextureInfo(texture).pixel_format));

    // MGL_TEXTURE_STORAGE_PRIVATE textures cannot be read directly with getBytes:.
    // Use a blit-to-buffer path to convert GPU-private tiled memory to linear CPU memory.
    if (mglTextureInfo(texture).storage_mode == MGL_TEXTURE_STORAGE_PRIVATE) {

        MGLRenderGetTexImagePlan plan = {0};
        mglRenderGetTexImagePlan(
            (uint32_t)mglTextureInfo(texture).pixel_format,
            (uint32_t)format,
            (uint32_t)type,
            (uint32_t)readRegion.size.width,
            (uint32_t)readRegion.size.height,
            (uint32_t)readRegion.size.depth,
            (uint32_t)dstPixelBytes,
            (uint32_t)mglMetalReadbackBytesPerPixel(mglTextureInfo(texture).pixel_format),
            mglMetalReadbackFormatIsBGRA8Compatible(mglTextureInfo(texture).pixel_format) ? 1 : 0,
            (uint32_t)bytesPerRow,
            (uint32_t)bytesPerImage,
            1,
            &plan);
        useBGRA8Conversion = plan.use_bgra8_conversion;
        NSUInteger rowBytes = (NSUInteger)plan.row_bytes;
        NSUInteger imageBytes = (NSUInteger)plan.image_bytes;
        NSUInteger totalBytes = (NSUInteger)plan.total_bytes;

        id stagingBuffer = mglTextureCreateBuffer(
            _device, totalBytes, MGL_TEXTURE_RESOURCE_STORAGE_SHARED);
        if (!stagingBuffer) {
            NSLog(@"MGL ERROR: mtlGetTexImage failed to allocate staging buffer for texture %u", tex->name);
            mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorOutOfMemory());
            return;
        }

        id blitCB = mglTextureCreateCommandBuffer(_commandQueue);
        if (!blitCB) {
            NSLog(@"MGL ERROR: mtlGetTexImage failed to create blit command buffer for texture %u", tex->name);
            mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
            return;
        }

        id blitEncoder = mglTextureCreateBlitEncoder(blitCB);
        if (!blitEncoder) {
            NSLog(@"MGL ERROR: mtlGetTexImage failed to create blit encoder for texture %u", tex->name);
            mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
            return;
        }

        mglTextureCopyTextureToBuffer(
            blitEncoder, texture, readSlice, level, readRegion.origin,
            readRegion.size, stagingBuffer, 0, rowBytes, imageBytes);

        mglTextureEndBlitEncoder(blitEncoder);
        mglTextureCommitCommandBuffer(blitCB);
        mglTextureWaitCommandBuffer(blitCB);

        MGLRenderCommandBufferState blitState = {0};
        (void)mglRenderGetCommandBufferState(
            (__bridge void *)blitCB, &blitState);
        if (blitState.has_error) {
            NSLog(@"MGL ERROR: mtlGetTexImage blit failed for texture %u: %s",
                  tex->name, mglRenderCommandBufferErrorDescription(&blitState));
            mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
            return;
        }

        if (useBGRA8Conversion) {
            if (!mglMetalCopyBGRA8CompatibleTextureBytesToGL((const uint8_t *)mglTextureBufferContents(stagingBuffer),
                                                             rowBytes,
                                                             (uint8_t *)pixelBytes,
                                                             bytesPerRow,
                                                             readRegion.size.width,
                                                             readRegion.size.height,
	                                                             mglTextureInfo(texture).pixel_format,
	                                                             format,
	                                                             type,
	                                                             flipRenderTargetRows)) {
	                NSLog(@"MGL ERROR: mtlGetTexImage unsupported BGRA8 conversion texture=%u format=0x%x type=0x%x",
	                      tex->name,
	                      (unsigned)format,
	                      (unsigned)type);
	                mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
	            }
        } else if (flipRenderTargetRows && readRegion.size.depth == 1u) {
            mglMetalCopyRows((const uint8_t *)mglTextureBufferContents(stagingBuffer),
                             rowBytes,
                             (uint8_t *)pixelBytes,
	                             bytesPerRow,
	                             rowBytes,
                             readRegion.size.height,
                             YES);
        } else {
            memcpy(pixelBytes, mglTextureBufferContents(stagingBuffer), totalBytes);
        }
        if (mglTraceLogIsEnabled() &&
            mglRenderTraceR8RedUByte((uint32_t)tex->internalformat,
                                     (uint32_t)format, (uint32_t)type) &&
            readRegion.size.width > 0 &&
            readRegion.size.height > 0) {
            const uint8_t *rb = (const uint8_t *)pixelBytes;
            mglTraceLog("GET_TEX_IMAGE_R8 tex=%u target=0x%x isRT=%d fmt=%lu rowBytes=%lu dstBPR=%lu size=%lux%lu first=%u,%u,%u,%u,%u,%u,%u,%u",
                        (unsigned)tex->name,
                        (unsigned)tex->target,
                        tex->is_render_target ? 1 : 0,
                        (unsigned long)mglTextureInfo(texture).pixel_format,
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

	            MGLRenderGetTexImagePlan plan = {0};
	            mglRenderGetTexImagePlan(
	                (uint32_t)mglTextureInfo(texture).pixel_format,
	                (uint32_t)format,
	                (uint32_t)type,
	                (uint32_t)readRegion.size.width,
	                (uint32_t)readRegion.size.height,
	                (uint32_t)readRegion.size.depth,
	                (uint32_t)dstPixelBytes,
	                (uint32_t)mglMetalReadbackBytesPerPixel(mglTextureInfo(texture).pixel_format),
	                mglMetalReadbackFormatIsBGRA8Compatible(mglTextureInfo(texture).pixel_format) ? 1 : 0,
	                (uint32_t)bytesPerRow,
	                (uint32_t)bytesPerImage,
	                0,
	                &plan);
	            NSUInteger rowBytes = (NSUInteger)plan.row_bytes;
	            NSUInteger totalBytes = (NSUInteger)plan.image_bytes;
            NSMutableData *readback = [NSMutableData dataWithLength:totalBytes];
            if (!readback) {
                mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorOutOfMemory());
                return;
            }
            mglTextureGetBytes(
                texture, readback.mutableBytes, rowBytes, bytesPerImage,
                readRegion, level, readSlice, YES);
            if (useBGRA8Conversion) {
                if (!mglMetalCopyBGRA8CompatibleTextureBytesToGL((const uint8_t *)readback.bytes,
                                                                 rowBytes,
                                                                 (uint8_t *)pixelBytes,
                                                                 bytesPerRow,
                                                                 readRegion.size.width,
                                                                 readRegion.size.height,
	                                                                 mglTextureInfo(texture).pixel_format,
	                                                                 format,
	                                                                 type,
	                                                                 flipRenderTargetRows)) {
	                    NSLog(@"MGL ERROR: mtlGetTexImage unsupported BGRA8 conversion texture=%u format=0x%x type=0x%x",
	                          tex->name,
	                          (unsigned)format,
	                          (unsigned)type);
                    mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
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
                readRegion, level, readSlice, YES);
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: mtlGetTexImage texture read failed for texture %u: %@",
              tex->name,
              exception);
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
    }
}

-(void)mtlGenerateMipmaps:(GLMContext)glm_ctx forTexture:(Texture *) tex
{
    ctx = glm_ctx;

    if (!tex) {
        NSLog(@"MGL ERROR: mtlGenerateMipmaps called with NULL texture");
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    RETURN_ON_FAILURE([self processGLState: false]);

    // end encoding on current render encoder
    [self endRenderEncoding];

    RETURN_ON_FAILURE([self ensureWritableCommandBuffer:"mtlGenerateMipmaps"]);

    // no failure path..?
    RETURN_ON_FAILURE([self bindMTLTexture:tex]);

    id texture;

    texture = (__bridge id)(tex->mtl_data);
    if (!texture) {
        NSLog(@"MGL ERROR: mtlGenerateMipmaps texture %u has no Metal texture after bind", tex->name);
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    if (mglTextureInfo(texture).mipmap_level_count <= 1u) {
        return;
    }

    // start blit encoder
    id blitCommandEncoder;
    blitCommandEncoder = mglTextureCreateCurrentBlitEncoder(
        _renderPassManager->state->currentCommandBufferOwner);
    if (!blitCommandEncoder) {
        NSLog(@"MGL ERROR: Failed to create blit encoder for mipmap generation");
        return;
    }

    @try {
        if (mglRenderBlitGenerateMipmaps(
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
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
    }
}

- (bool)encodeTextureBytesUpload:(Texture *)tex
                          source:(id)buffer
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

    id texture = (__bridge id)(tex->mtl_data);
    if (!texture) {
        return false;
    }

    uint32_t textureType = mglTextureInfo(texture).texture_type;
    MGLRenderTextureSubUploadPlan uploadPlan = {0};
    if (mglRenderTextureSubUploadPlan(
            (uint32_t)tex->target, (uint32_t)textureType, (uint64_t)slice,
            (uint64_t)xoffset, (uint64_t)yoffset, (uint64_t)zoffset,
            (uint64_t)width, (uint64_t)height, (uint64_t)depth,
            (uint64_t)sourceBytesPerRow, (uint64_t)sourceBytesPerImage,
            &uploadPlan) != 0) {
        return false;
    }
    NSUInteger destinationSlice = (NSUInteger)uploadPlan.destination_base_slice;
    MGLOriginValue destinationOrigin = mglTextureOrigin(
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
    if (textureType == MGLTextureTypeCube ||
        textureType == MGLTextureTypeCubeArray ||
        textureType == MGLTextureType2DArray ||
        textureType == MGLTextureType1DArray ||
        textureType == MGLTextureType2DMultisampleArray) {
        copyBytesPerImage = expectedBytesPerImage;
    } else if (textureType == MGLTextureType3D) {
        if (copyBytesPerImage < expectedBytesPerImage) {
            copyBytesPerImage = expectedBytesPerImage;
        }
    } else {
        copyBytesPerImage = expectedBytesPerImage;
    }

    NSUInteger maxDestinationSlices = mglTextureInfo(texture).array_length;
    if (textureType == MGLTextureTypeCube) {
        maxDestinationSlices = 6UL;
    } else if (textureType == MGLTextureTypeCubeArray) {
        maxDestinationSlices = mglTextureInfo(texture).array_length * 6UL;
    }

    if (level >= mglTextureInfo(texture).mipmap_level_count ||
        destinationSlice >= maxDestinationSlices ||
        layerCount > maxDestinationSlices - destinationSlice ||
        destinationOrigin.x > mglTextureInfo(texture).width ||
        destinationOrigin.y > mglTextureInfo(texture).height ||
        destinationOrigin.z > mglTextureInfo(texture).depth ||
        width > mglTextureInfo(texture).width - destinationOrigin.x ||
        copyHeight > mglTextureInfo(texture).height - destinationOrigin.y ||
        copyDepth > mglTextureInfo(texture).depth - destinationOrigin.z) {
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
              (unsigned long)mglTextureInfo(texture).width,
              (unsigned long)mglTextureInfo(texture).height,
              (unsigned long)mglTextureInfo(texture).depth);
        return false;
    }

    return mglTextureCopyUploadWithDedicatedCommandBuffer(
            (__bridge void *)self, (__bridge void *)buffer, sourceOffset, sourceBytesPerRow, copyBytesPerImage, sourceLayerStride, layerCount, mglTextureSize(
                                                       (NSUInteger)uploadPlan.copy_width,
                                                       copyHeight, copyDepth), (__bridge void *)texture, destinationSlice, level, destinationOrigin, reason ? reason : "texture_sub_upload");
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
        mglRendererBindMTLBuffer((__bridge void *)self, buf);
        RETURN_ON_NULL(buf->data.mtl_data);
    }

    id buffer = (__bridge id)(buf->data.mtl_data);
    if (!buffer) {
        NSLog(@"MGL ERROR: mtlTexSubImage missing Metal buffer object tex=%u", tex->name);
        return;
    }


    if (tex->mtl_data) {
        id dstTexture = (__bridge id)(tex->mtl_data);
        uint32_t dstPixelFormat = mglTextureInfo(dstTexture).pixel_format;
        BOOL needsChannelExpand = mglTextureNeedsChannelExpansion(tex->internalformat, dstPixelFormat);
        BOOL needsRGBA8Expand = NO;
        if (!needsChannelExpand) {
            needsRGBA8Expand = mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, dstPixelFormat);
        }
        if (needsChannelExpand || needsRGBA8Expand) {
            uint32_t rgbDst = 0u;
            NSUInteger dstBytesPerPixel = 4u;
            if (needsChannelExpand &&
                mglRenderRGBExpandParams(dstPixelFormat, NULL, &rgbDst,
                                         NULL)) {
                dstBytesPerPixel = (NSUInteger)rgbDst * 4u;
            } else if (needsChannelExpand) {
                dstBytesPerPixel = 16u;
            }
            NSUInteger cpuBytesPerPixel = (tex->faces[0].levels && level < tex->num_levels &&
                                           tex->faces[0].levels[level].width > 0u &&
                                           tex->faces[0].levels[level].pitch > 0u)
                ? (NSUInteger)(tex->faces[0].levels[level].pitch / tex->faces[0].levels[level].width)
                : mglTextureBytesPerPixelForFormat(tex->internalformat);
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
                    const uint8_t *sourceBase = (const uint8_t *)mglTextureBufferContents(buffer);
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
                            id uploadBuffer =
                                mglTextureCreateBufferWithBytes(
                                    _device, packedUpload.bytes, packedBytes,
                                    MGL_TEXTURE_RESOURCE_STORAGE_SHARED);
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

    NSUInteger bytesPerPixel = mglTextureBytesPerPixelForFormat(tex->internalformat);
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
    id dstTexture = (__bridge id)(tex->mtl_data);
    uint32_t dstPixelFormat = mglTextureInfo(dstTexture).pixel_format;
    BOOL needsChannelExpand = mglTextureNeedsChannelExpansion(tex->internalformat,
                                                              dstPixelFormat);
    NSUInteger dstBytesPerPixel = bytesPerPixel;
    if (needsChannelExpand) {
        uint32_t rgbDst = 0u;
        if (mglRenderRGBExpandParams(dstPixelFormat, NULL, &rgbDst, NULL)) {
            dstBytesPerPixel = (NSUInteger)rgbDst * 4u;
        } else {
            needsChannelExpand = NO;
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
        uint32_t srcCompU = 0u, dstCompU = 0u;
        uint64_t alphaDefault = 0;
        if (!mglRenderRGBExpandParams(dstPixelFormat, &srcCompU, &dstCompU,
                                      &alphaDefault)) {
            return false;
        }
        NSUInteger srcCompBytes = srcCompU;
        NSUInteger dstCompBytes = dstCompU;
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

    void *dsMetalUpload = NULL;
    const void *uploadBytesPtr = packedBytesPtr;
    NSUInteger uploadRowBytes = dstRowBytes;
    NSUInteger uploadImageBytes = dstImageBytes;
    if (mglRenderDepth32FStencil8NeedsUnpack(
            (uint32_t)tex->internalformat, (uint32_t)dstPixelFormat,
            (uint32_t)dstRowBytes, (uint32_t)width)) {
        NSUInteger expandedBPR = 0;
        NSUInteger expandedBPI = 0;
        dsMetalUpload = mglCreateDepthStencilMetalUpload(
            tex, dstPixelFormat, packedBytesPtr, width, copyHeight,
            dstRowBytes, &expandedBPR, &expandedBPI);
        if (dsMetalUpload) {
            uploadBytesPtr = dsMetalUpload;
            uploadRowBytes = expandedBPR;
            uploadImageBytes = expandedBPI;
        }
    }

    NSUInteger metalSlice = slice;
    if (mglRenderTextureTargetIsArray((uint32_t)tex->target)) {
        metalSlice = zoffset;
    }

    if (mglRenderPixelFormatIsPackedDepthStencil((uint32_t)dstPixelFormat) &&
        mglTextureInfo(dstTexture).storage_mode != MGL_TEXTURE_STORAGE_PRIVATE &&
        uploadRowBytes >= width * 5u) {
        bool uploaded = false;
        @try {
            mglTextureReplaceRegion(
                dstTexture,
                mglTextureRegion2D(xoffset, yoffset, width, copyHeight),
                level, metalSlice, uploadBytesPtr, uploadRowBytes,
                uploadImageBytes, YES);
            uploaded = true;
        } @catch (NSException *exception) {
            NSLog(@"MGL WARNING: depth/stencil texSubImage replaceRegion failed tex=%u: %@",
                  (unsigned)tex->name, exception.reason);
        }
        if (uploaded) {
            uploaded = mglTextureUploadPackedDepthStencilStencilPlane(
        (__bridge void *)dstTexture, tex->name, uploadBytesPtr, width, copyHeight, uploadRowBytes, level, metalSlice, xoffset, yoffset);
        }
        free(dsMetalUpload);
        return uploaded;
    }

    size_t uploadBufferBytes = uploadImageBytes * copyDepth;
    id uploadBuffer = mglTextureCreateBufferWithBytes(
        _device, uploadBytesPtr, uploadBufferBytes,
        MGL_TEXTURE_RESOURCE_STORAGE_SHARED);
    if (!uploadBuffer) {
        free(dsMetalUpload);
        return false;
    }

    bool uploaded = [self encodeTextureBytesUpload:tex
                                            source:uploadBuffer
                                      sourceOffset:0
                                  sourceBytesPerRow:uploadRowBytes
                                sourceBytesPerImage:uploadImageBytes
                                             width:width
                                            height:height
                                             depth:depth
                                             slice:slice
                                             level:level
                                           xoffset:xoffset
                                           yoffset:yoffset
                                           zoffset:zoffset
                                            reason:"mtlTexSubImageBytes"];
    if (uploaded &&
        mglRenderPixelFormatIsPackedDepthStencil((uint32_t)dstPixelFormat) &&
        uploadRowBytes >= width * 5u) {
        (void)mglTextureUploadPackedDepthStencilStencilPlane(
        (__bridge void *)dstTexture, tex->name, uploadBytesPtr, width, copyHeight, uploadRowBytes, level, metalSlice, xoffset, yoffset);
    }
    free(dsMetalUpload);
    if (uploaded && tex->is_render_target) {
        /* Direct CPU→Metal refresh of an FBO-attached texture must invalidate
         * the Y-flip sampled copy.  textures.c also releases the copy, but
         * bumping write_version keeps any concurrent/lazy refresh coherent
         * with the post-upload Metal contents (KHR-GL46.texture_barrier).
         * Rebuild immediately so every MRT attachment has a fresh Y-flip
         * copy before the first feedback draw (color1+ previously rebuilt
         * only as a side-effect of color0's sample-gate repair). */
        mglMarkTextureLevelMetalFilled(tex, level, packedBytes);
        id source = (__bridge id)(tex->mtl_data);
        if (source) {
            (void)mglBlitUpdateGLSampledRenderTargetCopy((__bridge void *)self, tex, (__bridge void *)source, "texSubImage_metal_fill");
        }
    }
    return uploaded;
}


#pragma mark - Extracted from createMTLTextureFromGLTexture:
- (void)reUploadExistingCPUTextureData:(Texture *)tex
                                metal:(id)texture
                          pixelFormat:(uint32_t)pixelFormat
                            numFaces:(uint)num_faces
                    uploadLevelCount:(GLuint)upload_level_count
                              isArray:(BOOL)is_array
                   texture1DBackedBy2D:(BOOL)texture1DBackedBy2D
             texture1DArrayBackedBy2DArray:(BOOL)texture1DArrayBackedBy2DArray
                             texType:(uint32_t)tex_type
{
    NSLog(@"MGL INFO: Re-uploading existing CPU texture data (tex=%d, dims=%lux%lu)",

          tex->name, (unsigned long)mglTextureInfo(texture).width, (unsigned long)mglTextureInfo(texture).height);


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


            BOOL is3DReupload = mglRenderIs3DReupload(
                                    (uint32_t)tex->target, (uint32_t)lvlDepth) != 0;

            NSUInteger singleSliceBPI = bytesPerRow * MAX((NSUInteger)lvlHeight, 1UL);

            NSUInteger bytesPerImage = is3DReupload ? singleSliceBPI : fullDataSize;

            NSUInteger uploadDepth = is3DReupload ? lvlDepth : (lvlDepth > 1 ? lvlDepth : 1);


            const void *srcData = (const void *)tex->faces[face].levels[level].data;

            void *expandedUploadData = NULL;

            /* Channel expansion for 2D/non-3D only.  3D expansion would

             * require per-slice handling (see DIRTY_TEXTURE_DATA 3D path). */

            if (!is3DReupload) {

                if (!mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) &&
                    mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, pixelFormat)) {

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

                } else if (!mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) &&
                           mglTextureNeedsChannelExpansion(tex->internalformat, pixelFormat)) {

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

            /* Combined depth/stencil CPU shadows use a packed layout
             * (DEPTH32F_STENCIL8 = 5 bytes/texel) while the Metal texture
             * expects 8 bytes/texel with stencil at byte 4; repack here so
             * the non-array refresh path matches the dirty/array paths. */
            NSUInteger dsBytesPerRow = 0;
            NSUInteger dsBytesPerImage = 0;
            void *dsUploadData = mglCreateDepthStencilMetalUpload(
                tex, pixelFormat, (const uint8_t *)srcData,
                lvlWidth, MAX((NSUInteger)lvlHeight, 1UL),
                bytesPerRow, &dsBytesPerRow, &dsBytesPerImage);
            if (dsUploadData) {
                free(expandedUploadData);
                expandedUploadData = dsUploadData;
                srcData = dsUploadData;
                bytesPerRow = dsBytesPerRow;
                bytesPerImage = dsBytesPerImage;
            }

            NSUInteger alignment = mglRendererOptimalAlignmentForPixelFormat(pixelFormat);

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

                        mglTextureUploadSliceViaBlit(
            (__bridge void *)self, (__bridge void *)texture, tex->name, tex->target, alignedData, alignedBytesPerRow, alignedSliceBPI, lvlWidth, lvlHeight, uploadDepth, level, face);

                        free(alignedData);

                    }

                }

            } else {

                mglTextureUploadSliceViaBlit(
            (__bridge void *)self, (__bridge void *)texture, tex->name, tex->target, srcData, bytesPerRow, bytesPerImage, lvlWidth, lvlHeight, uploadDepth, level, face);

            }

            free(expandedUploadData);

            } /* end else (non-array) */

        }

    }

}

- (void)fillTextureWithSafeInitialContents:(id)texture
                                         tex:(Texture *)tex
                                 pixelFormat:(uint32_t)pixelFormat
{


    if (mglTextureInfo(texture).width == 0 || mglTextureInfo(texture).height == 0 || mglTextureInfo(texture).width > 16384 || mglTextureInfo(texture).height > 16384) {

        NSLog(@"MGL WARNING: Skipping texture fill due to invalid dimensions: %lux%lu", (unsigned long)mglTextureInfo(texture).width, (unsigned long)mglTextureInfo(texture).height);

    } else {

        // Determine pixel format size to create appropriate black data

        NSUInteger bytesPerPixel = (NSUInteger)mglRenderMetalPixelFormatBytesPerPixel(
            mglTextureInfo(texture).pixel_format);

        // Calculate dynamic alignment for Metal textures based on pixel format

        NSUInteger bytesPerRow = mglTextureInfo(texture).width * bytesPerPixel;

        NSUInteger alignment = mglRendererOptimalAlignmentForPixelFormat(mglTextureInfo(texture).pixel_format);

        if (bytesPerRow % alignment != 0) {

            bytesPerRow = ((bytesPerRow + alignment - 1) / alignment) * alignment;

        }


        NSUInteger dataSize = bytesPerRow * mglTextureInfo(texture).height;


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

                if (mglTextureInfo(texture).width == 0 || mglTextureInfo(texture).height == 0) {

                    NSLog(@"MGL SECURITY ERROR: Invalid texture dimensions %lux%lu", (unsigned long)mglTextureInfo(texture).width, (unsigned long)mglTextureInfo(texture).height);

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

                if (mglTextureInfo(texture).width == 0 || mglTextureInfo(texture).height == 0) {

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

                NSUInteger properBytesPerRow = mglTextureInfo(texture).width * pixelSize;


                // Ensure proper alignment for Apple Metal driver

                if (properBytesPerRow % 64 != 0) {

                    properBytesPerRow = ((properBytesPerRow + 63) / 64) * 64;

                }


                // Fill the entire level. A previous 1x1 safety fill left large textures

                // mostly uninitialized while their Metal backing existed.

                MGLRegionValue properRegion = mglTextureRegion2D(0, 0, mglTextureInfo(texture).width, mglTextureInfo(texture).height);


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

                                id tempBuffer =
                                    mglTextureCreateBufferWithBytes(
                                        _device, properData, fillSize,
                                        MGL_TEXTURE_RESOURCE_STORAGE_SHARED);


                                if (tempBuffer) {

                                    NSLog(@"MGL INFO: Created temporary MTLBuffer for texture data");


                                    if (mglRendererShouldSkipGPUOperations((__bridge void *)self)) {

                                        NSLog(@"MGL AGX: Skipping texture fill during recovery - texture will be empty");

                                    } else {

                                        BOOL uploaded = mglTextureCopyUploadWithDedicatedCommandBuffer(
            (__bridge void *)self, (__bridge void *)tempBuffer, 0, properBytesPerRow, fillSize, 0, 1, mglTextureSize(properRegion.size.width, properRegion.size.height, 1), (__bridge void *)texture, 0, 0, mglTextureOrigin(0, 0, 0), "texture_fill_initialization");

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
                            metal:(id)texture
                      pixelFormat:(uint32_t)pixelFormat
                        numFaces:(uint)num_faces
                uploadLevelCount:(GLuint)upload_level_count
                         isArray:(BOOL)is_array
              texture1DBackedBy2D:(BOOL)texture1DBackedBy2D
        texture1DArrayBackedBy2DArray:(BOOL)texture1DArrayBackedBy2DArray
                         texType:(uint32_t)tex_type
            outAllLevelsUploaded:(BOOL *)outAllLevelsUploaded
{
    MGL_ASSERT_GL_THREAD();

    if (kMGLDiagnosticStateLogs) {
        mglTraceLog("MGL DEBUG: DIRTY_TEXTURE_DATA detected - attempting texture filling");
        mglTraceLog("MGL DEBUG: Texture details: target=0x%x, internalformat=0x%x, levels=%d effectiveLevels=%u",
                      tex->target, tex->internalformat, tex->num_levels, upload_level_count);
    }

    MGLRegionValue region;
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
                region = mglTextureRegion2D(0,0,width,1);
            else if (depth > 1)
                region = mglTextureRegion3D(0,0,0,width,height,depth);
            else if (height > 1)
                region = mglTextureRegion2D(0,0,width,height);
            else
                region = mglTextureRegion1D(0,width);

            NSUInteger bytesPerRow;
            NSUInteger bytesPerImage;
            bool hasExplicitDataSize = false;

            BOOL levelSkipped = NO;

            if (tex_type == MGLTextureType3D)
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
            else
                mglMarkGLSampledCopyLevelDirty(tex, (GLuint)level);
        }
    }

    if (outAllLevelsUploaded)
        *outAllLevelsUploaded = !anyLevelSkipped;

    return YES;
}

- (void)reUploadExistingCPUTextureDataArrayLevel:(Texture *)tex
                                          metal:(id)texture
                                    pixelFormat:(uint32_t)pixelFormat
                                          face:(int)face
                                         level:(int)level
                  texture1DArrayBackedBy2DArray:(BOOL)texture1DArrayBackedBy2DArray
                                       texType:(uint32_t)tex_type
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

                GLuint num_layers = (tex_type == MGLTextureType1DArray || texture1DArrayBackedBy2DArray)

                    ? tex->faces[face].levels[level].height

                    : tex->faces[face].levels[level].depth;

                if (num_layers == 0) return;


                BOOL arraySliceIs1D = (tex_type == MGLTextureType1DArray || texture1DArrayBackedBy2DArray);

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

                /* data_size is page-rounded; do not treat the slack as layer
                 * stride or reads land in the wrong slice. */

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
                    void *swizzledUploadData = NULL;

                    NSUInteger effectiveBytesPerRow = baseBytesPerRow;

                    NSUInteger effectiveBytesPerImage = logicalBytesPerImage;

                    if (mglTextureUploadNeedsSwizzleBake(tex)) {
                        NSUInteger swzBPR = 0;
                        NSUInteger swzBPI = 0;
                        swizzledUploadData = mglCreateSwizzledUpload(
                            tex, (const uint8_t *)layerSrcData, lvlWidth,
                            uploadSliceHeight, baseBytesPerRow, &swzBPR,
                            &swzBPI);
                        if (swizzledUploadData) {
                            layerSrcData = swizzledUploadData;
                            effectiveBytesPerRow = swzBPR;
                            effectiveBytesPerImage = swzBPI;
                        }
                    }

                    if (!mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) &&
                    mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, pixelFormat)) {

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

                    } else if (!mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) &&
                           mglTextureNeedsChannelExpansion(tex->internalformat, pixelFormat)) {

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

                    NSUInteger dsBytesPerRow = 0;
                    NSUInteger dsBytesPerImage = 0;
                    void *dsUploadData = mglCreateDepthStencilMetalUpload(
                        tex, pixelFormat, (const uint8_t *)layerSrcData,
                        lvlWidth, uploadSliceHeight, effectiveBytesPerRow,
                        &dsBytesPerRow, &dsBytesPerImage);
                    if (dsUploadData) {
                        free(expandedUploadData);
                        expandedUploadData = dsUploadData;
                        layerSrcData = dsUploadData;
                        effectiveBytesPerRow = dsBytesPerRow;
                        effectiveBytesPerImage = dsBytesPerImage;
                    }


                    NSUInteger alignment = mglRendererOptimalAlignmentForPixelFormat(pixelFormat);

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

                                mglTextureUploadSliceViaBlit(
            (__bridge void *)self, (__bridge void *)texture, tex->name, tex->target, alignedData, alignedBytesPerRow, alignedSize, lvlWidth, lvlHeight, 1, level, layer);

                                free(alignedData);

                            }

                        }

                    } else {

                        mglTextureUploadSliceViaBlit(
            (__bridge void *)self, (__bridge void *)texture, tex->name, tex->target, layerSrcData, effectiveBytesPerRow, effectiveBytesPerImage, lvlWidth, lvlHeight, 1, level, layer);

                    }

                    free(swizzledUploadData);
                    free(expandedUploadData);

                }

}

- (void)fillSmallRGBA8TextureWithGradient:(id)texture tex:(Texture *)tex
{
                            if (mglRenderIsSmallRGBA8(
                                    (uint32_t)mglTextureInfo(texture).width,
                                    (uint32_t)mglTextureInfo(texture).height,
                                    (uint32_t)tex->internalformat)) {

                                NSLog(@"MGL INFO: Attempting simple direct color fill for small RGBA8 texture");


                                @try {

                                    // Create a simple pattern that's not magenta

                                    NSUInteger pixelCount = mglTextureInfo(texture).width * mglTextureInfo(texture).height;

                                    uint32_t *simpleData = calloc(pixelCount, sizeof(uint32_t));


                                    if (simpleData) {

                                        // Create a simple gradient pattern instead of magenta

                                        for (NSUInteger y = 0; y < mglTextureInfo(texture).height; y++) {

                                            for (NSUInteger x = 0; x < mglTextureInfo(texture).width; x++) {

                                                NSUInteger index = y * mglTextureInfo(texture).width + x;


                                                // Create a simple gradient from blue to green

                                                uint8_t r = (uint8_t)(x * 255 / mglTextureInfo(texture).width);

                                                uint8_t g = (uint8_t)(y * 255 / mglTextureInfo(texture).height);

                                                uint8_t b = 128;

                                                uint8_t a = 255;


                                                simpleData[index] = (a << 24) | (b << 16) | (g << 8) | r;

                                            }

                                        }


                                        // Try direct replaceRegion for simple cases

                                        MGLRegionValue simpleRegion = mglTextureRegion2D(0, 0, mglTextureInfo(texture).width, mglTextureInfo(texture).height);

                                        mglTextureReplaceRegion(
                                            texture, simpleRegion, 0, 0,
                                            simpleData,
                                            mglTextureInfo(texture).width * sizeof(uint32_t),
                                            mglTextureInfo(texture).width * mglTextureInfo(texture).height * sizeof(uint32_t),
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
                                    metal:(id)texture
                              pixelFormat:(uint32_t)pixelFormat
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

                    uint8_t *swizzled3DUploadData = NULL;
                    if (level == 0 && face == 0 &&
                        mglTextureUploadNeedsSwizzleBake(tex)) {
                        NSUInteger texDepth = MAX((NSUInteger)depth, 1UL);
                        NSUInteger texHeight = MAX((NSUInteger)height, 1UL);
                        NSUInteger swzBPR = 0;
                        NSUInteger swzBPI = 0;
                        uint8_t *firstSlice =
                            mglCreateSwizzledUpload(
                                tex, (const uint8_t *)srcData, width, texHeight,
                                bytesPerRow, &swzBPR, &swzBPI);
                        if (firstSlice) {
                            NSUInteger totalSize = swzBPI * texDepth;
                            if (totalSize > 0 &&
                                totalSize <= (512 * 1024 * 1024)) {
                                swizzled3DUploadData =
                                    (uint8_t *)malloc(totalSize);
                                if (swizzled3DUploadData) {
                                    memcpy(swizzled3DUploadData, firstSlice,
                                           swzBPI);
                                    for (NSUInteger z = 1; z < texDepth; z++) {
                                        const uint8_t *sliceSrc =
                                            (const uint8_t *)srcData +
                                            z * bytesPerImage;
                                        uint8_t *sliceDst =
                                            swizzled3DUploadData + z * swzBPI;
                                        uint8_t *sliceSwz =
                                            mglCreateSwizzledUpload(
                                                tex, sliceSrc, width, texHeight,
                                                bytesPerRow, &swzBPR, &swzBPI);
                                        if (sliceSwz) {
                                            memcpy(sliceDst, sliceSwz, swzBPI);
                                            free(sliceSwz);
                                        } else {
                                            memset(sliceDst, 0, swzBPI);
                                        }
                                    }
                                    srcData = swizzled3DUploadData;
                                    bytesPerRow = swzBPR;
                                    bytesPerImage = swzBPI;
                                    addr = (uintptr_t)srcData;
                                }
                            }
                            free(firstSlice);
                        }
                    }

                    uint8_t *expanded3DUploadData = NULL;
                    if (!mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) &&
                    mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, pixelFormat)) {
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
                    } else if (!mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) &&
                           mglTextureNeedsChannelExpansion(tex->internalformat, pixelFormat)) {
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

                    NSUInteger alignment = mglRendererOptimalAlignmentForPixelFormat(pixelFormat);
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
                                BOOL uploaded = mglTextureUploadSliceViaBlit(
            (__bridge void *)self, (__bridge void *)texture, tex->name, tex->target, alignedData, alignedBytesPerRow, alignedBytesPerImage, width, height, depth, level, 0);
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
                            BOOL uploaded = mglTextureUploadSliceViaBlit(
            (__bridge void *)self, (__bridge void *)texture, tex->name, tex->target, srcData, bytesPerRow, bytesPerImage, width, height, depth, level, 0);
                            if (!uploaded) {
                                NSLog(@"MGL WARNING: 3D direct blit upload failed (level %d, face %d)", level, face);
                            }
                        } @catch (NSException *exception) {
                            NSLog(@"MGL ERROR: Failed to upload 3D texture data (level %d, face %d): %@", level, face, exception);
                        }
                    }
                    free(expanded3DUploadData);
                    free(swizzled3DUploadData);
                } else {
                    NSLog(@"MGL WARNING: Skipping 3D texture upload due to invalid data or parameters");
                }

    return YES;
}

- (BOOL)uploadDirtyCPUTextureDataNon3DLevel:(Texture *)tex
                                       metal:(id)texture
                                 pixelFormat:(uint32_t)pixelFormat
                                       face:(int)face
                                      level:(int)level
                                      width:(NSUInteger)width
                                     height:(NSUInteger)height
                                      depth:(NSUInteger)depth
                                   isArray:(BOOL)is_array
                  texture1DArrayBackedBy2DArray:(BOOL)texture1DArrayBackedBy2DArray
                                    texType:(uint32_t)tex_type
                                 outSkipped:(BOOL *)outSkipped
{
    NSUInteger bytesPerRow;
    NSUInteger bytesPerImage;
    bool hasExplicitDataSize = false;
    MGLRegionValue region;

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

                    num_layers = (tex_type == MGLTextureType1DArray || texture1DArrayBackedBy2DArray)
                        ? tex->faces[face].levels[level].height
                        : tex->faces[face].levels[level].depth;
                    if (num_layers == 0) {
                        NSLog(@"MGL WARNING: Array texture has 0 layers, skipping upload (tex=%d face=%d level=%d)", tex->name, face, level);
                        if (outSkipped) *outSkipped = YES;
                        return YES;
                    }

                    arraySliceIs1D = (tex_type == MGLTextureType1DArray || texture1DArrayBackedBy2DArray);
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
                    /* data_size is page-rounded; do not treat the slack as
                     * layer stride or reads land in the wrong slice. */
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
                        region = mglTextureRegion2D(0,0,width,height);
                    else if (height >= 1)
                        region = mglTextureRegion2D(0,0,width,1);
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
                            void *swizzledUploadData = NULL;
                            uintptr_t addr = (uintptr_t)srcData;

                            NSUInteger effectiveBytesPerRow = bytesPerRow;
                            NSUInteger effectiveBytesPerImage = bytesPerImage;
                            if (mglTextureUploadNeedsSwizzleBake(tex)) {
                                NSUInteger swzBPR = 0;
                                NSUInteger swzBPI = 0;
                                swizzledUploadData =
                                    mglCreateSwizzledUpload(
                                        tex, (const uint8_t *)srcData, width,
                                        uploadSliceHeight, bytesPerRow, &swzBPR,
                                        &swzBPI);
                                if (swizzledUploadData) {
                                    srcData = swizzledUploadData;
                                    effectiveBytesPerRow = swzBPR;
                                    effectiveBytesPerImage = swzBPI;
                                    addr = (uintptr_t)srcData;
                                }
                            }

                            if (!mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) &&
                    mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, pixelFormat)) {
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
                            } else if (!mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) &&
                           mglTextureNeedsChannelExpansion(tex->internalformat, pixelFormat)) {
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

                            NSUInteger dsBytesPerRow = 0;
                            NSUInteger dsBytesPerImage = 0;
                            void *dsUploadData = mglCreateDepthStencilMetalUpload(
                                tex, pixelFormat, (const uint8_t *)srcData,
                                width, uploadSliceHeight, effectiveBytesPerRow,
                                &dsBytesPerRow, &dsBytesPerImage);
                            if (dsUploadData) {
                                free(expandedUploadData);
                                expandedUploadData = dsUploadData;
                                srcData = dsUploadData;
                                effectiveBytesPerRow = dsBytesPerRow;
                                effectiveBytesPerImage = dsBytesPerImage;
                                addr = (uintptr_t)srcData;
                            }

                            NSUInteger alignment = mglRendererOptimalAlignmentForPixelFormat(pixelFormat);
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
                                    free(swizzledUploadData);
                                    free(expandedUploadData);
                                    continue;
                                }
                                NSUInteger alignedBytesPerImage = alignedBytesPerRow * alignedUploadRows;
                                NSUInteger alignedSize = alignedBytesPerImage;
                                if (alignedSize == 0 || alignedSize > (512 * 1024 * 1024)) {
                                    NSLog(@"MGL WARNING: Rejecting aligned array upload staging size=%lu (tex=%d level=%d layer=%d)",
                                          (unsigned long)alignedSize, tex->name, level, layer);
                                    free(swizzledUploadData);
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
                                            BOOL uploaded = mglTextureUploadSliceViaBlit(
            (__bridge void *)self, (__bridge void *)texture, tex->name, tex->target, alignedData, alignedBytesPerRow, alignedBytesPerImage, width, uploadSliceHeight, 1, level, layer);
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
                                    free(swizzledUploadData);
                                    free(expandedUploadData);
                                    continue;
                                }
                                if (effectiveBytesPerRow == 0) {
                                    NSLog(@"MGL SECURITY ERROR: Invalid bytesPerRow (0) passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash", level, layer);
                                    free(swizzledUploadData);
                                    free(expandedUploadData);
                                    continue;
                                }
                                if (effectiveBytesPerImage == 0) {
                                    NSLog(@"MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash", level, layer);
                                    free(swizzledUploadData);
                                    free(expandedUploadData);
                                    continue;
                                }
                                if (hasExplicitDataSize) {
                                    BOOL uploaded = mglTextureUploadSliceViaBlit(
            (__bridge void *)self, (__bridge void *)texture, tex->name, tex->target, srcData, effectiveBytesPerRow, effectiveBytesPerImage, width, uploadSliceHeight, 1, level, layer);
                                    if (!uploaded) {
                                        NSLog(@"MGL WARNING: Array texture direct blit upload failed (level %d, layer %d)", level, layer);
                                    }
                                } else {
                                    NSLog(@"MGL INFO: Skipping array upload with synthesized data size (level %d, layer %d)", level, layer);
                                }
                            }
                            free(swizzledUploadData);
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
                        if (level == 0 && face == 0 && mglTextureUploadNeedsSwizzleBake(tex)) {
                            NSUInteger swizzledBytesPerRow = 0;
                            NSUInteger swizzledBytesPerImage = 0;
                            swizzledUploadData = mglCreateSwizzledUpload(tex,
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
                            !mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) &&
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
                                   !mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) &&
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

                        NSUInteger alignment = mglRendererOptimalAlignmentForPixelFormat(pixelFormat);
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
                                    BOOL uploaded = mglTextureUploadSliceViaBlit(
            (__bridge void *)self, (__bridge void *)texture, tex->name, tex->target, alignedData, alignedBytesPerRow, alignedBytesPerImage, width, height, 1, level, face);
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
                                BOOL uploaded = mglTextureUploadSliceViaBlit(
            (__bridge void *)self, (__bridge void *)texture, tex->name, tex->target, srcData, bytesPerRow, bytesPerImage, width, height, 1, level, face);
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





- (id) createMTLTextureFromGLTexture:(Texture *) tex
{
    MGL_ASSERT_GL_THREAD();
    mglMetalCountCreate(MGLMetalKindTexture);
    // PROPER FIX: Enhanced pre-creation validation to prevent AGX driver issues
    if (!_device || !_commandQueue) {
        NSLog(@"MGL ERROR: Metal device or command queue not available for texture creation");
        return nil;
    }

    // Check if we're in a recovery state that would make texture creation futile
    if (mglRendererShouldSkipGPUOperations((__bridge void *)self)) {
        NSLog(@"MGL AGX: GPU operations temporarily suspended during recovery");
        return nil;
    }

    // Validate texture dimensions to prevent Metal assertion failures.
    // Texture buffers (GL_TEXTURE_BUFFER) can have very large widths (millions of texels)
    // since they map to MGLTextureTypeTextureBuffer which uses GPU address space.
    if (!mglRenderTextureDimsValid(tex->target, tex->width, tex->height,
                                   tex->depth)) {
            NSLog(@"MGL ERROR: Invalid texture dimensions %dx%dx%d - rejecting",
                  tex ? tex->width : 0, tex ? tex->height : 0, tex ? tex->depth : 0);
            tex->dirty_bits = 0;
            return nil;
        }

    if (mglRenderIsTextureBufferTarget(tex->target)) {
        return (__bridge id)mglTextureCreateMTLTexelBufferTexture(
            (__bridge void *)self, tex);
    }

    NSUInteger width, height, depth;

    MGLRenderTextureDescriptorState tex_desc = {0};
    uint32_t tex_type;
    uint32_t pixelFormat;
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

    MGLRenderTextureTargetPlan targetPlan = {0};
    if (mglRenderTextureTargetPlan(
            (uint32_t)tex->target,
            (uint32_t)tex->samples,
            &targetPlan) != 0) {
        NSLog(@"MGL TEXTURE ERROR: unsupported texture target 0x%x for Metal texture creation tex=%u",
              tex->target,
              tex->name);
        return nil;
    }
    tex_type = (uint32_t)targetPlan.texture_type;
    num_faces = (uint)targetPlan.num_faces;
    is_array = targetPlan.is_array != 0u;
    texture1DBackedBy2D = targetPlan.texture_1d_backed_by_2d != 0u;
    texture1DArrayBackedBy2DArray =
        targetPlan.texture_1d_array_backed_by_2d_array != 0u;

    int effectiveMipmapped = 0;
    if (!mglTextureCheckCompleteness(tex, tex_type, num_faces,
                                     &effective_mipmap_levels,
                                     &effectiveMipmapped)) {
        return nil;
    }
    storageMipmapped = effectiveMipmapped ? YES : NO;

    // PROPER FIX: Get original texture format and validate for AGX compatibility
    pixelFormat = mtlPixelFormatForGLTex(tex);
    BOOL expandsSingleChannelSwizzle = mglTextureUploadNeedsSingleChannelSwizzle(tex);
    BOOL usesUploadSwizzleBake = mglTextureUploadNeedsSwizzleBake(tex);
    pixelFormat = mglRenderResolveUploadSwizzlePixelFormat(
        pixelFormat, expandsSingleChannelSwizzle ? 1 : 0,
        mglRenderSingleChannelSwizzleStoragePixelFormat(
            (uint32_t)tex->internalformat),
        mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) ? 1 : 0,
        mglRenderIntegerMultiChannelSwizzleStoragePixelFormat(
            (uint32_t)tex->internalformat),
        mglTextureUploadNeedsStencilSwizzleBake(tex) ? 1 : 0,
        mglRenderStencilSwizzleStoragePixelFormat(),
        mglTextureUploadNeedsDepthStencilDepthSwizzleBake(tex) ? 1 : 0,
        mglRenderSingleChannelSwizzleStoragePixelFormat(
            (uint32_t)tex->internalformat));

    // Validate format compatibility with AGX, but preserve original intent
    BOOL needsFormatConversion = NO;
    uint32_t originalFormat = pixelFormat;
    int agxConverted = 0;
    pixelFormat = mglRenderAGXCompatiblePixelFormat(pixelFormat, &agxConverted);
    needsFormatConversion = agxConverted != 0;

    /* Metal does not allow depth/stencil pixel formats with MGLTextureType1DArray.
     * Promote to MGLTextureType2DArray with height=1, mirroring how mipmapped
     * 1D array textures are already promoted below.  Without this, creating a
     * GL_TEXTURE_1D_ARRAY depth texture (e.g. sampler_1d_array_shadow) triggers
     * a Metal validation assertion crash. */
    if (mglRenderPromote1DArrayDepthStencil(tex_type, pixelFormat)) {
        tex_type = MGLTextureType2DArray;
        texture1DArrayBackedBy2DArray = true;
    }

    width = tex->width;
    height = tex->height;
    depth = tex->depth;
    if (tex_type == MGLTextureType2DMultisample ||
        tex_type == MGLTextureType2DMultisampleArray) {
        storageMipmapped = NO;
        effective_mipmap_levels = 1u;
        tex->mipmapped = false;
    }

    mipmapped = storageMipmapped;
    /* GL may allocate num_levels>1 for a single-base-level image; only walk
     * mips that were actually populated unless the texture is mipmapped. */
    upload_level_count = mglRenderUploadLevelCount(
        mipmapped ? 1 : 0, tex->mipmapped ? 1 : 0, effective_mipmap_levels);

    tex_desc.texture_type = tex_type;
    tex_desc.pixel_format = pixelFormat;
    tex_desc.width = width;
    tex_desc.height = mglRenderTextureDescHeight(tex_type, (uint32_t)height);
    bool msEmulatedAsArray = false;
    {
        uint32_t outType = tex_type;
        uint32_t sampleCount = 1u;
        uint64_t arrayLen = 1u;
        uint64_t descDepth = 1u;
        NSUInteger samples = MAX((NSUInteger)2u, (NSUInteger)tex->samples);
        samples = MGLCapabilityClampSampleCount(&_capability, samples);
        if (mglRenderEmulateMSAsArray(tex_type, (uint32_t)samples,
                                      (uint64_t)depth, &outType, &sampleCount,
                                      &arrayLen, &descDepth)) {
            msEmulatedAsArray = true;
            tex_type = outType;
            tex_desc.texture_type = tex_type;
            tex_desc.sample_count = sampleCount;
            tex_desc.array_length = arrayLen;
            tex_desc.depth = descDepth;
        }
    }

    // CONSERVATIVE: Use only Metal API patterns that work reliably with AGX driver
    tex_desc.cpu_cache_mode = MGLCapabilityUseConservativeCPUCache(&_capability)
        ? MGL_TEXTURE_CPU_CACHE_WRITE_COMBINED
        : MGL_TEXTURE_CPU_CACHE_DEFAULT;

    // Use shared storage for textures that need CPU upload (blit/replaceRegion).
    // Private storage is only safe for pure GPU render targets on Apple Silicon.
    bool hasUploadableCPUData = mglTextureHasUploadableCPUData(tex, num_faces, upload_level_count);
    bool needsCpuUpload = ((tex->dirty_bits & DIRTY_TEXTURE_DATA) != 0) && hasUploadableCPUData;
    bool preferSharedDepthStencil =
        mglMetalPixelFormatIsDepthOrStencil(pixelFormat);
    tex_desc.storage_mode =
        mglRenderPreferSharedStorage(needsCpuUpload ? 1 : 0,
                                     preferSharedDepthStencil ? 1 : 0)
            ? 0u
            : MGL_TEXTURE_STORAGE_PRIVATE;
    tex_desc.sample_count = MAX(tex_desc.sample_count, 1u);
    tex_desc.mipmap_level_count = MAX(tex_desc.mipmap_level_count, 1u);
    tex_desc.array_length = MAX(tex_desc.array_length, 1u);
    tex_desc.depth = MAX(tex_desc.depth, 1u);

    // Normalize depth/array semantics per Metal texture type.
    if ((tex_type == MGLTextureTypeCube ||
         tex_type == MGLTextureTypeCubeArray) &&
        !mglRenderCubeFaceSizeValid((uint64_t)width, (uint64_t)height)) {
            NSLog(@"MGL ERROR: invalid cube texture size %lux%lu for tex=%u glTarget=0x%x",
                  (unsigned long)width, (unsigned long)height, tex->name, tex->target);
    }
    if (tex_type == MGLTextureTypeCubeArray) {
        uint64_t cubeCount = (uint64_t)depth;
        if (cubeCount > 1u && (cubeCount % 6u) != 0u) {
            NSLog(@"MGL WARNING: cube-array depth=%lu is not a multiple of 6, treating as cube count",
                  (unsigned long)cubeCount);
        }
    }
    uint64_t arrayLen = tex_desc.array_length;
    uint64_t descDepth = tex_desc.depth;
    if (mglRenderTextureArrayDepthForType(
            tex_type, is_array ? 1 : 0, msEmulatedAsArray ? 1 : 0,
            (uint64_t)width, (uint64_t)height, (uint64_t)depth, &arrayLen,
            &descDepth)) {
        tex_desc.array_length = arrayLen;
        tex_desc.depth = descDepth;
    }

    if (mipmapped)
    {
        if (mglRenderPromoteMipmapped1D(tex_type)) {
            tex_type = MGLTextureType2D;
            texture1DBackedBy2D = true;
        }
        /* Metal does not allow mipmapLevelCount > 1 for MGLTextureType1DArray.
         * Promote to MGLTextureType2DArray with height=1 to support mipmapped
         * 1D array textures.  The upload code checks texture1DArrayBackedBy2DArray
         * to treat each slice as 1 pixel tall. */
        if (mglRenderPromoteMipmapped1DArray(tex_type)) {
            tex_type = MGLTextureType2DArray;
            texture1DArrayBackedBy2DArray = true;
        }
        tex_desc.mipmap_level_count = MAX((GLuint)1, effective_mipmap_levels);
    }

    if (texture1DBackedBy2D || texture1DArrayBackedBy2DArray) {
        uint32_t backedType = tex_desc.texture_type;
        uint64_t backedArray = tex_desc.array_length;
        uint32_t backedHeight = (uint32_t)tex_desc.height;
        mglRenderApply1DBackingToDesc(texture1DBackedBy2D ? 1 : 0,
                                      texture1DArrayBackedBy2DArray ? 1 : 0,
                                      (uint64_t)height, &backedType,
                                      &backedArray, &backedHeight);
        tex_desc.texture_type = backedType;
        tex_desc.array_length = backedArray;
        tex_desc.height = backedHeight;
    }

    /* GL image access mode (GL_READ_ONLY / GL_WRITE_ONLY / GL_READ_WRITE)
     * only governs the image binding, NOT the texture's overall capabilities.
     * A texture bound as a write-only image may still be sampled from via
     * sampler2D in the same shader.  Metal requires MGL_TEXTURE_USAGE_SHADER_READ
     * for sampling, so always include it alongside the image write flag.
     *
     * AIR always declares storage images as access::read_write (see
     * mgl_air_backend.cpp).  Binding a ShaderRead-only texture to that slot
     * yields zeroed imageLoad results on AGX, so READ_ONLY also needs
     * ShaderWrite even though GLSL/GL mark the binding readonly. */
    uint32_t accessUsage = 0u;
    if (!mglRenderTextureUsageForAccess((uint32_t)tex->access, &accessUsage)) {
            NSLog(@"MGL TEXTURE ERROR: invalid texture access 0x%x for tex=%u",
                  tex->access,
                  tex->name);
            return nil;
    }
    tex_desc.usage = accessUsage;

    /* Metal 3.1 imageAtomic* requires ShaderAtomic on R32{U,S}int textures. */
    if (mglRenderPixelFormatNeedsShaderAtomic(pixelFormat)) {
        tex_desc.usage |= MGL_TEXTURE_USAGE_SHADER_ATOMIC;
    }

    if (tex->is_render_target)
    {
        tex_desc.usage |= MGL_TEXTURE_USAGE_RENDER_TARGET | MGL_TEXTURE_USAGE_SHADER_READ;
    }

    // Allow safe same-memory format reinterpretation (e.g. RGBA8 <-> BGRA8)
    // for blit/present paths where OpenGL attachments and drawable formats differ.
    tex_desc.usage |= MGL_TEXTURE_USAGE_PIXEL_FORMAT_VIEW;

    if (tex_desc.texture_type == MGLTextureTypeCube || tex_desc.texture_type == MGLTextureTypeCubeArray) {
        NSLog(@"MGL CUBE DESC tex=%u glTarget=0x%x type=%lu width=%lu height=%lu depth=%lu arrayLength=%lu pixelFormat=%lu usage=%lu storage=%lu mipmapped=%d",
              tex->name,
              tex->target,
              (unsigned long)tex_desc.texture_type,
              (unsigned long)tex_desc.width,
              (unsigned long)tex_desc.height,
              (unsigned long)tex_desc.depth,
              (unsigned long)tex_desc.array_length,
              (unsigned long)tex_desc.pixel_format,
              (unsigned long)tex_desc.usage,
              (unsigned long)tex_desc.storage_mode,
              (int)mipmapped);
    }

    if (tex->params.swizzled && !usesUploadSwizzleBake &&
        !tex->is_render_target)
    {
        mglTextureSwizzleDescriptor(&tex_desc, tex);
    }

    id texture;

    // CRITICAL FIX: Safe texture creation with proper validation
    @try {
        texture = mglTextureCreateTexture(_device, &tex_desc);
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception creating texture: %@", exception);
        mglRendererRecordGPUError((__bridge void *)self);
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

    if (cpuUploadRequired && mglRenderTextureTargetIs2D((uint32_t)tex->target) &&
        mglTextureInfo(texture).texture_type == MGLTextureType2D &&
        !mglTextureUploadNeedsSwizzleBake(tex)) {
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

    mglRendererRecordGPUSuccess((__bridge void *)self);

    return texture;
}

- (void)flushImageUnitSlice:(GLMContext)glm_ctx unit:(GLuint)unit
{
    if (!glm_ctx || unit >= glm_ctx->active_state->var.max_image_units ||
        unit >= TEXTURE_UNITS) {
        return;
    }
    ImageUnit *iu = &glm_ctx->active_state->image_units[unit];
    if (!mglRenderImageUnitSliceNeedsFlush(
            iu->tex ? 1 : 0, iu->mtl_image_view ? 1 : 0, iu->layered ? 1 : 0,
            iu->tex ? (uint32_t)iu->tex->target : 0u,
            (uint32_t)iu->access)) {
        return;
    }
    if (![self bindMTLTexture:iu->tex] || !iu->tex->mtl_data) {
        return;
    }
    id dst3d = (__bridge id)iu->tex->mtl_data;
    id staging = (__bridge id)iu->mtl_image_view;
    MGLRenderTextureInfo info = mglTextureInfo(dst3d);
    if (info.texture_type != MGLTextureType3D || info.width == 0u) {
        return;
    }
    const NSUInteger level = (NSUInteger)iu->level;
    const NSUInteger layer = (NSUInteger)iu->layer;
    if (level >= info.mipmap_level_count || layer >= info.depth) {
        return;
    }

    [self endRenderEncoding];
    if (!_renderPassManager->state->currentCommandBufferOwner &&
        ![self newCommandBufferLocked]) {
        return;
    }
    void *blit = mglRenderCreateBlitEncoderBorrowed(
        _renderPassManager->state->currentCommandBufferOwner);
    if (!blit) {
        return;
    }
    (void)mglRenderBlitCopyTexture(
        blit, (__bridge void *)staging, 0u, 0u, 0u, 0u, 0u,
        info.width, info.height, 1u,
        (__bridge void *)dst3d, 0u, level, 0u, 0u, layer);
    (void)mglRenderEndBlitEncoder(blit);
    [self flushCommandBuffer:NO];
}

- (void)prepareImageUnitSlice:(GLMContext)glm_ctx unit:(GLuint)unit
{
    if (!glm_ctx || unit >= glm_ctx->active_state->var.max_image_units ||
        unit >= TEXTURE_UNITS) {
        return;
    }
    ImageUnit *iu = &glm_ctx->active_state->image_units[unit];
    if (!iu->tex || iu->layered ||
        !mglRenderTextureTargetIs3D((uint32_t)iu->tex->target)) {
        return;
    }
    if (![self bindMTLTexture:iu->tex] || !iu->tex->mtl_data) {
        return;
    }
    id src3d = (__bridge id)iu->tex->mtl_data;
    MGLRenderTextureInfo info = mglTextureInfo(src3d);
    if (info.texture_type != MGLTextureType3D || info.width == 0u) {
        return;
    }
    const NSUInteger level = (NSUInteger)iu->level;
    const NSUInteger layer = (NSUInteger)iu->layer;
    if (level >= info.mipmap_level_count || layer >= info.depth) {
        return;
    }

    if (iu->mtl_image_view) {
        [self flushImageUnitSlice:glm_ctx unit:unit];
        mglRenderReleaseMetalObject(iu->mtl_image_view);
        iu->mtl_image_view = NULL;
    }

    MGLRenderTextureDescriptorState desc = {
        .texture_type = MGLTextureType2D,
        .pixel_format = info.pixel_format,
        .width = info.width,
        .height = info.height,
        .depth = 1u,
        .mipmap_level_count = 1u,
        .sample_count = 1u,
        .array_length = 1u,
        .usage = MGL_TEXTURE_USAGE_SHADER_READ | MGL_TEXTURE_USAGE_SHADER_WRITE |
                 MGL_TEXTURE_USAGE_PIXEL_FORMAT_VIEW,
        .storage_mode = info.storage_mode,
    };
    id staging = mglTextureCreateTexture(_device, &desc);
    if (!staging) {
        return;
    }

    [self endRenderEncoding];
    if (!_renderPassManager->state->currentCommandBufferOwner &&
        ![self newCommandBufferLocked]) {
        return;
    }
    void *blit = mglRenderCreateBlitEncoderBorrowed(
        _renderPassManager->state->currentCommandBufferOwner);
    if (!blit) {
        return;
    }
    (void)mglRenderBlitCopyTexture(
        blit, (__bridge void *)src3d, 0u, level, 0u, 0u, layer,
        info.width, info.height, 1u,
        (__bridge void *)staging, 0u, 0u, 0u, 0u, 0u);
    (void)mglRenderEndBlitEncoder(blit);
    [self flushCommandBuffer:NO];

    iu->mtl_image_view = (__bridge_retained void *)staging;
}

- (void)syncTextureBufferFromImage:(GLMContext)glm_ctx tex:(Texture *)tex
{
    if (!glm_ctx || !tex ||
        !mglRenderTextureTargetIsBuffer((uint32_t)tex->target) ||
        !tex->mtl_data || !tex->texture_buffer || tex->texture_buffer_size <= 0) {
        return;
    }

    Buffer *sourceBuffer = tex->texture_buffer;
    id texture = (__bridge id)(tex->mtl_data);
    MGLRenderTextureInfo info = mglTextureInfo(texture);
    if (info.width == 0u || info.height == 0u) {
        return;
    }

    NSUInteger bytesPerTexel = mglTextureBytesPerPixelForFormat(tex->internalformat);
    if (bytesPerTexel == 0u) {
        bytesPerTexel = (NSUInteger)sizeForInternalFormat(tex->internalformat, 0, 0);
    }
    if (bytesPerTexel == 0u) {
        return;
    }

    NSUInteger bytesPerRow = (NSUInteger)info.width * bytesPerTexel;
    NSUInteger packedBytes = bytesPerRow * (NSUInteger)info.height;
    if (packedBytes == 0u ||
        (size_t)tex->texture_buffer_size > packedBytes) {
        return;
    }

    NSMutableData *packedData = [NSMutableData dataWithLength:packedBytes];
    if (!packedData.mutableBytes) {
        return;
    }

    @try {
        mglTextureGetBytes(
            texture, packedData.mutableBytes, bytesPerRow, 0,
            mglTextureRegion2D(0, 0, info.width, info.height), 0, 0, NO);
    } @catch (NSException *exception) {
        NSLog(@"MGL TEXBUFFER SYNC ERROR: getBytes failed tex=%u buffer=%u: %@",
              tex->name, sourceBuffer->name, exception);
        return;
    }

    mglRendererBufferSubData(
        glm_ctx, sourceBuffer,
        (size_t)tex->texture_buffer_offset,
        (size_t)tex->texture_buffer_size,
        packedData.bytes);
}

- (void)logMTLTextureMipDiagnostics:(Texture *)tex
                              metal:(id)texture
               effectiveMipLevels:(GLuint)effective_mipmap_levels
{
    static uint64_t s_mipDiagLogs = 0;
    uint64_t diagHit = ++s_mipDiagLogs;
    if (kMGLDiagnosticStateLogs &&
        (diagHit <= 128ull || (diagHit % 512ull) == 0ull)) {
        NSUInteger mtlMipCount = mglTextureInfo(texture).mipmap_level_count;
        uint32_t mtlFmt = mglTextureInfo(texture).pixel_format;
        uint32_t mtlStorage = mglTextureInfo(texture).storage_mode;
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
        mglTraceLog("MGL TEX_MIP_DIAG tex=%u target=0x%x dims=%ux%u internal=0x%x "
                      "numLevels=%u mipmapLevels=%u effectiveMipLevels=%u mtlMipCount=%lu "
                      "mtlFmt=%lu mtlStorage=%ld mipmapped=%d baseLevel=%u maxLevel=%u "
                      "uploadedLevels=%lu skippedLevels=%lu skippedSourceNone=%lu skippedNoData=%lu "
                      "levels=%s hit=%llu",
                      (unsigned)tex->name, (unsigned)tex->target,
                      (unsigned)tex->width, (unsigned)tex->height,
                      (unsigned)tex->internalformat,
                      (unsigned)tex->num_levels, (unsigned)tex->mipmap_levels,
                      (unsigned)effective_mipmap_levels, (unsigned long)mtlMipCount,
                      (unsigned long)mtlFmt, (long)mtlStorage, (int)(tex->mipmapped ? 1 : 0),
                      (unsigned)tex->params.base_level, (unsigned)tex->params.max_level,
                      (unsigned long)uploadedLevels, (unsigned long)skippedLevels,
                      (unsigned long)skippedSourceNone, (unsigned long)skippedNoData,
                      [levelSummary UTF8String], (unsigned long long)diagHit);
    }
}

// AGX-SAFE Fallback texture creation for GPU error recovery scenarios
- (id) createFallbackMTLTexture:(Texture *) tex
{
    // Validate texture parameters before creating Metal texture to prevent Metal assertion failures
    if (!tex || tex->width <= 0 || tex->height <= 0 || tex->width > 32768 || tex->height > 32768) {
        NSLog(@"MGL AGX: Skipping fallback texture creation - invalid dimensions %dx%d",
              tex ? tex->width : 0, tex ? tex->height : 0);
        return nil;
    }

    NSLog(@"MGL AGX: Creating emergency fallback texture (size: %dx%dx%d)", tex->width, tex->height, tex->depth);

    @try {
        uint32_t fallbackFormat = mglRenderFallbackPixelFormat(
            mtlPixelFormatForGLTex(tex), (uint32_t)tex->internalformat);

        BOOL isDepthOrStencilFormat =
            mglRenderPixelFormatIsDepthOrStencil(fallbackFormat) != 0;

        MGLRenderTextureDescriptorState fallbackDesc = {
            .texture_type = MGLTextureType2D,
            .pixel_format = fallbackFormat,
            .width = MAX(tex->width, 1), .height = MAX(tex->height, 1),
            .depth = 1u, .mipmap_level_count = 1u,
            .sample_count = 1u, .array_length = 1u,
            .usage = MGL_TEXTURE_USAGE_SHADER_READ,
        };
        if (tex->is_render_target || isDepthOrStencilFormat) {
            fallbackDesc.usage |= MGL_TEXTURE_USAGE_RENDER_TARGET;
        }

        id fallbackTexture =
            mglTextureCreateTexture(_device, &fallbackDesc);

        if (fallbackTexture) {
            // Fill with simple gradient pattern using a simple approach
            NSUInteger width = mglTextureInfo(fallbackTexture).width;
            NSUInteger height = mglTextureInfo(fallbackTexture).height;

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

                    MGLRegionValue region = mglTextureRegion2D(0, 0, width, height);
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





- (void)traceSampledTextureReadback:(id)texture
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

    MGLRenderTextureInfo textureInfo = {0};
    if (mglRenderGetTextureInfo((__bridge void *)texture,
                                   &textureInfo) != 0) {
        return;
    }
    uint32_t fmt = textureInfo.pixel_format;
    BOOL fourByteColor =
        mglRenderPixelFormatIsUnorm8Color(fmt) != 0;
    if (!fourByteColor) {
        mglTraceLog("MGL TRACE sampled.readback skip program=%u binding=%u glTex=%u reason=%s fmt=%lu type=%lu size=%lux%lu hit=%llu",
              (unsigned)program,
              (unsigned)binding,
              glTex ? (unsigned)glTex->name : 0u,
              [reason UTF8String],
              (unsigned long)fmt,
              (unsigned long)textureInfo.texture_type,
              (unsigned long)textureInfo.width,
              (unsigned long)textureInfo.height,
              (unsigned long long)hit);
        return;
    }

    NSUInteger texWidth = (NSUInteger)textureInfo.width;
    NSUInteger texHeight = (NSUInteger)textureInfo.height;
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

    id readback = mglTextureCreateBuffer(
        _device, byteCount, MGL_TEXTURE_RESOURCE_STORAGE_SHARED);
    id cb = mglTextureCreateCommandBuffer(_commandQueue);
    id blit = mglTextureCreateBlitEncoder(cb);
    if (!readback || !cb || !blit) {
        mglTraceLog("MGL TRACE sampled.readback setup-fail program=%u binding=%u glTex=%u reason=%s readback=%p cb=%p blit=%p hit=%llu",
              (unsigned)program,
              (unsigned)binding,
              glTex ? (unsigned)glTex->name : 0u,
              [reason UTF8String],
              readback,
              cb,
              blit,
              (unsigned long long)hit);
        return;
    }

    mglTextureCopyTextureToBuffer(
        blit, texture, 0, 0, mglTextureOrigin(0, 0, 0),
        mglTextureSize(sampleWidth, sampleHeight, 1), readback, 0,
        bytesPerRow, byteCount);
    mglTextureEndBlitEncoder(blit);
    mglTextureCommitCommandBuffer(cb);
    mglTextureWaitCommandBuffer(cb);

    const uint8_t *p = (const uint8_t *)mglTextureBufferContents(readback);
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

    MGLRenderCommandBufferState sampledState = {0};
    (void)mglRenderGetCommandBufferState(
        (__bridge void *)cb, &sampledState);
    NSString *sampledError = sampledState.has_error
        ? [NSString stringWithFormat:@"%s (domain=%s code=%lld)",
             sampledState.error_description,
             sampledState.error_domain,
             (long long)sampledState.error_code]
        : nil;
    mglTraceLog("MGL TRACE sampled.readback stage=%s program=%u binding=%u glTex=%u reason=%s hit=%llu "
          "mtl=%p fmt=%lu type=%lu size=%lux%lu sample=%lux%lu status=%s error=%@ "
          "nonZero=%lu/%lu sum=%llu first=0x%08x min=0x%08x max=0x%08x xor=0x%08x "
          "level(init ever=%u full=%u zero=%u source=%u upload=%lu src=%p hash=0x%016llx)",
          [stage UTF8String],
          (unsigned)program,
          (unsigned)binding,
          glTex ? (unsigned)glTex->name : 0u,
          reason,
          (unsigned long long)hit,
          texture,
          (unsigned long)fmt,
          (unsigned long)textureInfo.texture_type,
          (unsigned long)texWidth,
          (unsigned long)texHeight,
          (unsigned long)sampleWidth,
          (unsigned long)sampleHeight,
          mglCommandBufferStatusName(
              (uint32_t)sampledState.status),
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
