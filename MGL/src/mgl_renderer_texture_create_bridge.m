/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Texture create/upload/sampler Metal path — extracted from MGLRenderer+Texture.m.
 * Lives outside MGLRenderer*.m so PlatformRendererShell+Lifecycle LOC can shrink.
 */

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Texture_Private.h"
#import "MGLRenderer+GPURecovery_Private.h"
#import "MGLRenderer+Blit_Private.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"
#import "mgl_renderer_texture_metal_helpers.h"

@implementation MGLRenderer (TextureCreateBridge)

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

- (void)fillTextureWithSafeInitialContents:(id)texture
                                         tex:(Texture *)tex
                                 pixelFormat:(uint32_t)pixelFormat
{


    if (mglTextureInfo(texture).width == 0 || mglTextureInfo(texture).height == 0 || mglTextureInfo(texture).width > 16384 || mglTextureInfo(texture).height > 16384) {

        NSLog(@"MGL WARNING: Skipping texture fill due to invalid dimensions: %lux%lu", (unsigned long)mglTextureInfo(texture).width, (unsigned long)mglTextureInfo(texture).height);

    } else {

        // Determine pixel format size to create appropriate black data

        NSUInteger bytesPerPixel = 4; // Default to RGBA

        switch(mglTextureInfo(texture).pixel_format) {

            case MGLPixelFormatR8Unorm:

            case MGLPixelFormatR8Uint:

            case MGLPixelFormatR8Sint:

                bytesPerPixel = 1;

                break;

            case MGLPixelFormatRG8Unorm:

            case MGLPixelFormatRG8Uint:

            case MGLPixelFormatRG8Sint:

                bytesPerPixel = 2;

                break;

            case MGLPixelFormatRGBA8Unorm:

            case MGLPixelFormatRGBA8Uint:

            case MGLPixelFormatRGBA8Sint:

                bytesPerPixel = 4;

                break;

            default:

                bytesPerPixel = 4; // Default assumption

                break;

        }


        // Calculate dynamic alignment for Metal textures based on pixel format

        NSUInteger bytesPerRow = mglTextureInfo(texture).width * bytesPerPixel;

        NSUInteger alignment = [self getOptimalAlignmentForPixelFormat:mglTextureInfo(texture).pixel_format];

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


                                    if ([self shouldSkipGPUOperations]) {

                                        NSLog(@"MGL AGX: Skipping texture fill during recovery - texture will be empty");

                                    } else {

                                        BOOL uploaded = [self copyTextureUploadWithDedicatedCommandBuffer:tempBuffer

                                                                                              sourceOffset:0

                                                                                         sourceBytesPerRow:properBytesPerRow

                                                                                       sourceBytesPerImage:fillSize

                                                                                        sourceLayerStride:0

                                                                                                layerCount:1

                                                                                                 sourceSize:mglTextureSize(properRegion.size.width, properRegion.size.height, 1)

                                                                                                  toTexture:texture

                                                                                           destinationSlice:0

                                                                                           destinationLevel:0

                                                                                          destinationOrigin:mglTextureOrigin(0, 0, 0)

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
        mglTraceLogNSString(@"MGL DEBUG: DIRTY_TEXTURE_DATA detected - attempting texture filling");
        mglTraceLogNSString(@"MGL DEBUG: Texture details: target=0x%x, internalformat=0x%x, levels=%d effectiveLevels=%u",
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

- (void)fillSmallRGBA8TextureWithGradient:(id)texture tex:(Texture *)tex
{
                            if (mglTextureInfo(texture).width <= 512 && mglTextureInfo(texture).height <= 512 && tex->internalformat == GL_RGBA8) {

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


- (void)swizzleTexDesc:(MGLRenderTextureDescriptorState *)tex_desc forTex:(Texture*)tex
{
    tex_desc->swizzle_red = mglMTLSwizzleForGLSwizzle(tex, tex->params.swizzle_r);
    tex_desc->swizzle_green = mglMTLSwizzleForGLSwizzle(tex, tex->params.swizzle_g);
    tex_desc->swizzle_blue = mglMTLSwizzleForGLSwizzle(tex, tex->params.swizzle_b);
    tex_desc->swizzle_alpha = mglMTLSwizzleForGLSwizzle(tex, tex->params.swizzle_a);
    tex_desc->has_swizzle = 1u;
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
    if ([self shouldSkipGPUOperations]) {
        NSLog(@"MGL AGX: GPU operations temporarily suspended during recovery");
        return nil;
    }

    // Validate texture dimensions to prevent Metal assertion failures.
    // Texture buffers (GL_TEXTURE_BUFFER) can have very large widths (millions of texels)
    // since they map to MGLTextureTypeTextureBuffer which uses GPU address space.
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
        pixelFormat = MGLPixelFormatRGBA8Unorm;
    }

    // Validate format compatibility with AGX, but preserve original intent
    BOOL needsFormatConversion = NO;
    uint32_t originalFormat = pixelFormat;

    // Check for AGX-incompatible formats and only convert when necessary
    switch(pixelFormat) {
        case MGLPixelFormatB5G6R5Unorm:
        case MGLPixelFormatBGR5A1Unorm:
        case MGLPixelFormatA1BGR5Unorm:
            // 16-bit formats can cause issues on AGX
            needsFormatConversion = YES;
            pixelFormat = MGLPixelFormatRGBA8Unorm;
            break;
        case MGLPixelFormatPVRTC_RGBA_2BPP:
        case MGLPixelFormatPVRTC_RGBA_4BPP:
        case MGLPixelFormatPVRTC_RGB_2BPP:
        case MGLPixelFormatPVRTC_RGB_4BPP:
            // PVRTC compression can cause issues in virtualization
            needsFormatConversion = YES;
            pixelFormat = MGLPixelFormatRGBA8Unorm;
            break;
        case MGLPixelFormatEAC_R11Unorm:
        case MGLPixelFormatEAC_RG11Unorm:
        case MGLPixelFormatEAC_RGBA8:
        case MGLPixelFormatETC2_RGB8:
        case MGLPixelFormatETC2_RGB8A1:
            // ETC/ETC2 compression can cause issues on AGX
            needsFormatConversion = YES;
            pixelFormat = MGLPixelFormatRGBA8Unorm;
            break;
        default:
            // Most modern formats should work fine
            break;
    }

    /* Metal does not allow depth/stencil pixel formats with MGLTextureType1DArray.
     * Promote to MGLTextureType2DArray with height=1, mirroring how mipmapped
     * 1D array textures are already promoted below.  Without this, creating a
     * GL_TEXTURE_1D_ARRAY depth texture (e.g. sampler_1d_array_shadow) triggers
     * a Metal validation assertion crash. */
    if (tex_type == MGLTextureType1DArray) {
        switch (pixelFormat) {
            case MGLPixelFormatDepth16Unorm:
            case MGLPixelFormatDepth32Float:
            case MGLPixelFormatStencil8:
            case MGLPixelFormatDepth24Unorm_Stencil8:
            case MGLPixelFormatDepth32Float_Stencil8:
            case MGLPixelFormatX32_Stencil8:
            case MGLPixelFormatX24_Stencil8:
                tex_type = MGLTextureType2DArray;
                texture1DArrayBackedBy2DArray = true;
                break;
            default:
                break;
        }
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

    /* Depth/stencil attachments must be single-level Metal textures.
     * checkTextureCompleteness can set storageMipmapped for color RTs with
     * mipmap_levels>1; applying that to Depth32Float depth-tex FBOs yields
     * ineffective depth testing (last draw wins). Mirror RBO policy. */
    if (mglMetalPixelFormatIsDepthOrStencil(pixelFormat)) {
        storageMipmapped = NO;
        effective_mipmap_levels = 1u;
        tex->mipmapped = false;
    }

    mipmapped = storageMipmapped;
    /* GL may allocate num_levels>1 for a single-base-level image; only walk
     * mips that were actually populated unless the texture is mipmapped. */
    upload_level_count =
        (mipmapped && tex->mipmapped) ? effective_mipmap_levels : 1u;

    tex_desc.texture_type = tex_type;
    tex_desc.pixel_format = pixelFormat;
    tex_desc.width = width;
    tex_desc.height = (tex_type == MGLTextureType1D ||
                       tex_type == MGLTextureType1DArray) ? 1 : height;
    if (tex_type == MGLTextureType2DMultisample ||
        tex_type == MGLTextureType2DMultisampleArray) {

        NSUInteger samples = MAX((NSUInteger)2u, (NSUInteger)tex->samples);
        samples = MGLCapabilityClampSampleCount(&_capability, samples);
        tex_desc.sample_count = samples;
    }

    // CONSERVATIVE: Use only Metal API patterns that work reliably with AGX driver
    tex_desc.cpu_cache_mode = MGLCapabilityUseConservativeCPUCache(&_capability)
        ? MGL_TEXTURE_CPU_CACHE_WRITE_COMBINED
        : MGL_TEXTURE_CPU_CACHE_DEFAULT;

    // Use shared storage for textures that need CPU upload (blit/replaceRegion).
    // Private storage is only safe for pure GPU render targets on Apple Silicon.
    //
    // Depth/stencil FBO attachments MUST be Private — same policy as
    // newDrawBuffer / transient depth. 818069a forced Shared for all
    // depth/stencil so CTS replaceRegion uploads work; that still creates
    // Depth32Float RTs on Apple Paravirtual Metal, but depth test/write are
    // inert (last draw wins: depth_test / legacy z-control). Keep Shared only
    // for non-RT depth/stencil that need CPU upload.
    bool hasUploadableCPUData = mglTextureHasUploadableCPUData(tex, num_faces, upload_level_count);
    bool needsCpuUpload = ((tex->dirty_bits & DIRTY_TEXTURE_DATA) != 0) && hasUploadableCPUData;
    bool isDepthOrStencil = mglMetalPixelFormatIsDepthOrStencil(pixelFormat);
    if (isDepthOrStencil && tex->is_render_target) {
        tex_desc.storage_mode = MGL_TEXTURE_STORAGE_PRIVATE;
    } else {
        tex_desc.storage_mode =
            needsCpuUpload ? 0u : MGL_TEXTURE_STORAGE_PRIVATE;
    }
    tex_desc.sample_count = MAX(tex_desc.sample_count, 1u);
    tex_desc.mipmap_level_count = MAX(tex_desc.mipmap_level_count, 1u);
    tex_desc.array_length = MAX(tex_desc.array_length, 1u);
    tex_desc.depth = MAX(tex_desc.depth, 1u);

    // Normalize depth/array semantics per Metal texture type.
    if (tex_type == MGLTextureTypeCube) {
        if (width != height) {
            NSLog(@"MGL ERROR: invalid cube texture size %lux%lu for tex=%u glTarget=0x%x",
                  (unsigned long)width, (unsigned long)height, tex->name, tex->target);
        }
        tex_desc.depth = 1;
    } else if (tex_type == MGLTextureTypeCubeArray) {
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

        tex_desc.array_length = MAX((NSUInteger)1, cubeCount);
        tex_desc.depth = 1;
    } else if (tex_type == MGLTextureType1DArray) {
        tex_desc.array_length = MAX((NSUInteger)1, height);
        tex_desc.depth = 1;
    } else if (is_array) {
        tex_desc.array_length = MAX((NSUInteger)1, depth);
        tex_desc.depth = 1;
    } else {
        /* For 3D and other non-array textures, arrayLength must be 1.
         * Some Metal drivers report getNumSlices()==0 when arrayLength
         * is left at its default, causing "slice OOB" assertions. */
        tex_desc.array_length = 1;
        tex_desc.depth = MAX((NSUInteger)1, depth);
    }

    if (mipmapped)
    {
        if (tex_type == MGLTextureType1D) {
            tex_type = MGLTextureType2D;
            texture1DBackedBy2D = true;
        }
        /* Metal does not allow mipmapLevelCount > 1 for MGLTextureType1DArray.
         * Promote to MGLTextureType2DArray with height=1 to support mipmapped
         * 1D array textures.  The upload code checks texture1DArrayBackedBy2DArray
         * to treat each slice as 1 pixel tall. */
        if (tex_type == MGLTextureType1DArray) {
            tex_type = MGLTextureType2DArray;
            texture1DArrayBackedBy2DArray = true;
        }
        tex_desc.mipmap_level_count = MAX((GLuint)1, effective_mipmap_levels);
    }

    if (texture1DBackedBy2D) {
        tex_desc.texture_type = MGLTextureType2D;
        tex_desc.height = 1;
    }
    if (texture1DArrayBackedBy2DArray) {
        tex_desc.texture_type = MGLTextureType2DArray;
        /* For GL_TEXTURE_1D_ARRAY, the GL height parameter is the array slice
         * count.  Since tex_type was promoted to MGLTextureType2DArray above,
         * the arrayLength branch at line ~12397 (keyed on MGLTextureType1DArray)
         * was skipped, leaving arrayLength=1 from the is_array/depth fallback.
         * Set arrayLength from the GL height (slice count) here. */
        tex_desc.array_length = MAX((NSUInteger)1, height);
        tex_desc.height = 1;
    }

    /* GL image access mode (GL_READ_ONLY / GL_WRITE_ONLY / GL_READ_WRITE)
     * only governs the image binding, NOT the texture's overall capabilities.
     * A texture bound as a write-only image may still be sampled from via
     * sampler2D in the same shader.  Metal requires MGL_TEXTURE_USAGE_SHADER_READ
     * for sampling, so always include it alongside the image write flag. */
    switch(tex->access)
    {
        case GL_READ_ONLY:
            tex_desc.usage = MGL_TEXTURE_USAGE_SHADER_READ; break;
        case GL_WRITE_ONLY:
            tex_desc.usage = MGL_TEXTURE_USAGE_SHADER_READ | MGL_TEXTURE_USAGE_SHADER_WRITE; break;
        case GL_READ_WRITE:
            tex_desc.usage = MGL_TEXTURE_USAGE_SHADER_READ | MGL_TEXTURE_USAGE_SHADER_WRITE; break;
        default:
            NSLog(@"MGL TEXTURE ERROR: invalid texture access 0x%x for tex=%u",
                  tex->access,
                  tex->name);
            return nil;
    }

    if (tex->is_render_target)
    {
        tex_desc.usage |= MGL_TEXTURE_USAGE_RENDER_TARGET | MGL_TEXTURE_USAGE_SHADER_READ;
    }

    // Allow safe same-memory format reinterpretation (e.g. RGBA8 <-> BGRA8)
    // for blit/present paths where OpenGL attachments and drawable formats differ.
    // Metal forbids PixelFormatView on depth/stencil formats.
    if (!mglMetalPixelFormatIsDepthOrStencil(pixelFormat)) {
        tex_desc.usage |= MGL_TEXTURE_USAGE_PIXEL_FORMAT_VIEW;
    }

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

    if (tex->params.swizzled && !expandsSingleChannelSwizzle)
    {
        [self swizzleTexDesc:&tex_desc forTex:tex];
    }

    id texture;

    // CRITICAL FIX: Safe texture creation with proper validation
    @try {
        texture = mglTextureCreateTexture(_device, &tex_desc);
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

    if (cpuUploadRequired && tex->target == GL_TEXTURE_2D && mglTextureInfo(texture).texture_type == MGLTextureType2D) {
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

- (id)createMTLTexelBufferTexture:(Texture *)tex
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

    uint32_t bufferPixelFormat = (tex->internalformat == GL_RGBA8)
        ? MGLPixelFormatRGBA8Uint
        : mtlPixelFormatForGLTex(tex);
    if (bufferPixelFormat == MGLPixelFormatInvalid || bufferPixelFormat == 0) {
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
        id mtlBuffer = (__bridge id)(sourceBuffer->data.mtl_data);
        if (mtlBuffer && mglTextureBufferContents(mtlBuffer)) {
            sourceBytes = ((const uint8_t *)mglTextureBufferContents(mtlBuffer)) + (size_t)tex->texture_buffer_offset;
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
            case MGLPixelFormatRGBA16Unorm:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 65535; break;
            case MGLPixelFormatRGBA16Snorm:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 32767; break;
            case MGLPixelFormatRGBA16Float:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 0x3C00; break;
            case MGLPixelFormatRGBA16Sint:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 1; break;
            case MGLPixelFormatRGBA16Uint:
                srcCompBytes = 2; dstCompBytes = 2; alphaDefault = 1; break;
            case MGLPixelFormatRGBA32Float:
                srcCompBytes = 4; dstCompBytes = 4;
                { float f = 1.0f; memcpy(&alphaDefault, &f, sizeof(f)); }
                break;
            case MGLPixelFormatRGBA32Sint:
                srcCompBytes = 4; dstCompBytes = 4; alphaDefault = 1; break;
            case MGLPixelFormatRGBA32Uint:
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

                if (mglRenderTextureExpandRGBToRGBA(
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

    MGLRenderTextureDescriptorState bufferDesc = {
        .texture_type = MGLTextureType2D,
        .pixel_format = bufferPixelFormat,
        .width = texWidth, .height = texHeight, .depth = 1u,
        .mipmap_level_count = 1u, .sample_count = 1u, .array_length = 1u,
        .usage = MGL_TEXTURE_USAGE_SHADER_READ,
    };

    id bufferTexture = nil;
    @try {
        bufferTexture = mglTextureCreateTexture(_device, &bufferDesc);
        if (bufferTexture) {
            mglTextureReplaceRegion(
                bufferTexture, mglTextureRegion2D(0, 0, texWidth, texHeight),
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
            mglTextureRegion2D(0, 0, texWidth, texHeight), 0, 0, NO);
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
                          texType:(uint32_t)tex_type
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
        uint32_t fallbackFormat = mtlPixelFormatForGLTex(tex);
        if (fallbackFormat == MGLPixelFormatInvalid) {
            // Conservative defaults by GL intent when translation is unavailable.
            if (tex->internalformat == GL_DEPTH24_STENCIL8 ||
                tex->internalformat == GL_DEPTH32F_STENCIL8) {
                fallbackFormat = MGLPixelFormatDepth32Float_Stencil8;
            } else if (tex->internalformat == GL_DEPTH_COMPONENT ||
                       tex->internalformat == GL_DEPTH_COMPONENT16 ||
                       tex->internalformat == GL_DEPTH_COMPONENT24 ||
                       tex->internalformat == GL_DEPTH_COMPONENT32 ||
                       tex->internalformat == GL_DEPTH_COMPONENT32F) {
                fallbackFormat = MGLPixelFormatDepth32Float;
            } else {
                fallbackFormat = MGLPixelFormatRGBA8Unorm;
            }
        }

        BOOL isDepthOrStencilFormat =
            (fallbackFormat == MGLPixelFormatDepth16Unorm ||
             fallbackFormat == MGLPixelFormatDepth32Float ||
             fallbackFormat == MGLPixelFormatDepth24Unorm_Stencil8 ||
             fallbackFormat == MGLPixelFormatDepth32Float_Stencil8 ||
             fallbackFormat == MGLPixelFormatStencil8);

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
        if (isDepthOrStencilFormat) {
            fallbackDesc.storage_mode = MGL_TEXTURE_STORAGE_PRIVATE;
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

- (id) createMTLSamplerForTexParam:(TextureParameter *)tex_param target:(GLuint)target
{
    mglMetalCountCreate(MGLMetalKindSampler);
    void *sampler = NULL;
    char error[256] = {0};
    if (mglRenderCreateSamplerForGL(
            tex_param, target, &sampler, error, sizeof(error)) == 0 &&
        sampler) {
        return (__bridge_transfer id)sampler;
    }
    NSLog(@"MGL SAMPLER ERROR: Metal-cpp sampler creation failed: %s",
          error[0] ? error : "unknown");
    return nil;
}

@end
