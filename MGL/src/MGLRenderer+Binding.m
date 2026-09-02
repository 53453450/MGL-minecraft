/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+Binding.m
// Buffer/texture Metal object binding methods extracted from MGLRenderer+RenderPass.m

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Blit_Private.h"
#include "mgl_render.h"

void mglRendererCompatBindTexture(GLMContext glm_ctx,
                                  Texture *texture)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx || !texture) return;
    (void)[renderer bindMTLTexture:texture];
}

static id mglBindingCreateDefaultSampler(void)
{
    void *sampler = NULL;
    if (mglRenderCreateDefaultSampler(&sampler) == 0 && sampler) {
        return (__bridge_transfer id)sampler;
    }
    return nil;
}

@implementation MGLRenderer (Binding)

- (void) bindMTLBuffer:(Buffer *) ptr
{
    METAL_LOCK();
    [self bindMTLBufferLocked:ptr];
    METAL_UNLOCK();
}

- (void) bindMTLBufferLocked:(Buffer *)ptr
{
    char bindError[256] = {0};
    int bindResult = mglRenderBindBufferStorage(
        ptr, bindError, sizeof(bindError));
    if (bindResult != MGL_RENDER_BUFFER_BOUND) {
        NSLog(@"MGL ERROR: Metal-cpp buffer bind failed buffer=%u: %s",
              ptr ? (unsigned)ptr->name : 0u,
              bindError[0] ? bindError : "?");
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

    if (tex->mtl_data &&
        (tex->target == GL_TEXTURE_2D_ARRAY ||
         tex->target == GL_TEXTURE_CUBE_MAP_ARRAY ||
         tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY)) {
        MGLRenderTextureInfo existingInfo = {0};
        if (mglRenderGetTextureInfo(tex->mtl_data, &existingInfo) == 0) {
            const uint64_t expectedLayers = MAX((uint64_t)tex->depth, 1u);
            if (existingInfo.array_length < expectedLayers ||
                existingInfo.width != (uint64_t)tex->width ||
                existingInfo.height != (uint64_t)tex->height) {
                tex->dirty_bits |= DIRTY_TEXTURE_LEVEL | DIRTY_TEXTURE_DATA;
            }
        }
    }

    // If this texture is now used as a render target but was previously created
    // without render-target usage, force a recreate with proper usage flags.
    // When the old texture already has GPU-written data (e.g. from imageStore
    // in a compute shader), preserve it via a GPU-to-GPU blit instead of
    // re-uploading potentially stale CPU data.
    if (tex->mtl_data && tex->is_render_target) {
        id existingTexture = (__bridge id)(tex->mtl_data);
        MGLRenderTextureInfo existingInfo = {0};
        BOOL hasExistingInfo = existingTexture &&
            mglRenderGetTextureInfo((__bridge void *)existingTexture,
                                       &existingInfo) == 0;
        if (existingTexture && !hasExistingInfo) {
            NSLog(@"MGL ERROR: Failed to query texture %u metadata before render-target transition",
                  tex->name);
            return false;
        }
        const uint64_t requiredRenderTargetUsage = (1ull << 2) | (1ull << 0);
        NSUInteger requiredMipLevels =
            (tex->target == GL_RENDERBUFFER || tex->samples > 1u)
                ? 1u
                : ((tex->mipmap_levels > 1u) ? (NSUInteger)tex->mipmap_levels : 1u);
        BOOL usageMismatch = hasExistingInfo &&
            ((existingInfo.usage & requiredRenderTargetUsage) != requiredRenderTargetUsage);
        BOOL mipCountMismatch = hasExistingInfo &&
            requiredMipLevels > existingInfo.mipmap_level_count;
        if (existingTexture && (usageMismatch || mipCountMismatch)) {
            NSLog(@"MGL WARNING: Recreating texture %u for render-target use (old usage=0x%lx oldMips=%lu requiredMips=%lu)",
                  tex->name,
                  (unsigned long)existingInfo.usage,
                  (unsigned long)existingInfo.mipmap_level_count,
                  (unsigned long)requiredMipLevels);

            // Keep a strong reference to the old texture so we can blit its GPU
            // data to the new one after releasing tex->mtl_data.
            __strong id oldTexture = existingTexture;

            mglSafeReleaseMetalObj((void **)&tex->mtl_data);
            [self releaseGLSampledRenderTargetCopyForTexture:tex];

            // Create a new texture with correct usage.  Don't set
            // DIRTY_TEXTURE_DATA so that createMTLTextureFromGLTexture
            // skips CPU data upload — we'll blit GPU data instead.
            id newTexture = [self createMTLTextureFromGLTexture:tex];
            MGLRenderTextureInfo newInfo = {0};
            BOOL dimensionsMatch = newTexture &&
                mglRenderGetTextureInfo((__bridge void *)newTexture,
                                           &newInfo) == 0 &&
                newInfo.width == existingInfo.width &&
                newInfo.height == existingInfo.height &&
                newInfo.depth == existingInfo.depth;
            if (oldTexture && dimensionsMatch) {
                tex->mtl_data = (void *)CFBridgingRetain(newTexture);
                const BOOL packedDepthStencil =
                    tex->internalformat == GL_DEPTH32F_STENCIL8 ||
                    tex->internalformat == GL_DEPTH24_STENCIL8;
                if (packedDepthStencil) {
                    const BOOL isArray =
                        tex->target == GL_TEXTURE_2D_ARRAY ||
                        tex->target == GL_TEXTURE_CUBE_MAP_ARRAY ||
                        tex->target == GL_TEXTURE_1D_ARRAY ||
                        tex->target == GL_TEXTURE_CUBE_MAP ||
                        tex->target == GL_TEXTURE_3D;
                    BOOL allLevelsUploaded = YES;
                    const GLuint levelCount = (GLuint)MIN(
                        newInfo.mipmap_level_count,
                        tex->num_levels ? tex->num_levels : 1u);
                    if ([self uploadDirtyCPUTextureData:tex
                                                   metal:newTexture
                                             pixelFormat:(uint32_t)newInfo.pixel_format
                                               numFaces:1
                                       uploadLevelCount:levelCount
                                                isArray:isArray
                                     texture1DBackedBy2D:(tex->target == GL_TEXTURE_1D)
                               texture1DArrayBackedBy2DArray:(tex->target == GL_TEXTURE_1D_ARRAY)
                                                texType:(uint32_t)newInfo.texture_type
                                    outAllLevelsUploaded:&allLevelsUploaded] &&
                        allLevelsUploaded) {
                        tex->dirty_bits &= ~DIRTY_TEXTURE_DATA;
                    } else {
                        tex->dirty_bits |= DIRTY_TEXTURE_DATA;
                    }
                } else {
                    // Blit GPU data from old texture to new texture to preserve
                    // any writes (e.g. imageStore) that occurred before the
                    // is_render_target transition.
                    [self endRenderEncodingLocked];
                    if ([self ensureWritableCommandBufferLocked:"is_render_target_blit"]) {
                        if (mglRenderCopyMatchingTextureSubresourcesForCommandBufferOwner(
                                _renderPassManager.state->currentCommandBufferOwner,
                                (__bridge void *)oldTexture,
                                (__bridge void *)newTexture) != 0) {
                            NSLog(@"MGL ERROR: Metal-cpp render-target preservation blit failed texture=%u",
                                  tex->name);
                            tex->dirty_bits |= DIRTY_TEXTURE_DATA;
                            return false;
                        }
                    }
                    tex->dirty_bits = 0;
                }
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
            id existingTexture = (__bridge id)(tex->mtl_data);
            BOOL uploadedDirty = NO;
            if (existingTexture) {
                if (tex->target == GL_TEXTURE_2D) {
                    MGLRenderTextureInfo metalInfo = {0};
                    if (mglRenderGetTextureInfo((__bridge void *)existingTexture,
                                                   &metalInfo) == 0 &&
                        metalInfo.texture_type == MGLTextureType2D &&
                        !mglTextureUploadNeedsSwizzleBake(tex)) {
                        uploadedDirty =
                            [self uploadFullCPUTextureDataIntoTexture:tex
                                                                metal:existingTexture
                                                               reason:"bindMTLTexture.dirtyData"];
                    }
                }
                if (!uploadedDirty) {
                    MGLRenderTextureInfo metalInfo = {0};
                    if (mglRenderGetTextureInfo((__bridge void *)existingTexture,
                                                   &metalInfo) == 0) {
                        const BOOL isArray =
                            tex->target == GL_TEXTURE_2D_ARRAY ||
                            tex->target == GL_TEXTURE_CUBE_MAP_ARRAY ||
                            tex->target == GL_TEXTURE_1D_ARRAY ||
                            tex->target == GL_TEXTURE_CUBE_MAP ||
                            tex->target == GL_TEXTURE_3D;
                        const BOOL texture1DBackedBy2D =
                            tex->target == GL_TEXTURE_1D;
                        const BOOL texture1DArrayBackedBy2DArray =
                            tex->target == GL_TEXTURE_1D_ARRAY;
                        BOOL allLevelsUploaded = YES;
                        const GLuint levelCount = (GLuint)MIN(
                            metalInfo.mipmap_level_count,
                            tex->num_levels ? tex->num_levels : 1u);
                        uploadedDirty =
                            [self uploadDirtyCPUTextureData:tex
                                                       metal:existingTexture
                                                 pixelFormat:(uint32_t)metalInfo.pixel_format
                                                   numFaces:1
                                           uploadLevelCount:levelCount
                                                    isArray:isArray
                                         texture1DBackedBy2D:texture1DBackedBy2D
                                   texture1DArrayBackedBy2DArray:texture1DArrayBackedBy2DArray
                                                    texType:(uint32_t)metalInfo.texture_type
                                        outAllLevelsUploaded:&allLevelsUploaded] &&
                            allLevelsUploaded;
                    }
                }
            }
            if (uploadedDirty) {
                tex->dirty_bits &= ~DIRTY_TEXTURE_DATA;
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
                mglBindingCreateDefaultSampler());
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
        id mtlTex = (__bridge id)(tex->mtl_data);
        MGLRenderTextureInfo textureInfo = {0};
        BOOL hasTextureInfo = mtlTex &&
            mglRenderGetTextureInfo((__bridge void *)mtlTex, &textureInfo) == 0;
        uint64_t signature = 1469598103934665603ULL;
        signature = mglMipDiagMixState(signature, (uint64_t)(uintptr_t)tex->mtl_data);
        signature = mglMipDiagMixState(
            signature, hasTextureInfo ? textureInfo.mipmap_level_count : 0u);
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
                  (unsigned long)(hasTextureInfo ? textureInfo.mipmap_level_count : 0u),
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
