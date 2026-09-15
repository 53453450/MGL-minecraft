/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * MGLRenderer+Texture_Private.h
 * MGL
 *
 * Private method declarations, constants, and C helpers for the Texture
 * category (MGLRenderer+Texture.m).  Imports MGLRenderer_Private.h for
 * ivar access and shared types.
 */

#ifndef MGLRenderer_Texture_Private_h
#define MGLRenderer_Texture_Private_h

#import "MGLRenderer.h"

/* Value geometry types + constructors now live in mgl_region_value.h
 * (C++-safe, O4 dedup sink).  MGL_VALUE_GEOMETRY_TYPES guard is defined there. */
#include "mgl_region_value.h"

/* === Texture upload diagnostic constants === */
static const BOOL kMGLSynchronizeTextureUploads = NO;
static const NSTimeInterval kMGLTextureUploadWaitTimeoutSeconds = 0.25;
static const BOOL kMGLUseDedicatedTextureUploadCommandBuffer = NO;

/* === C functions defined in MGLRenderer.m, used by MGLRenderer+Texture.m === */
void mglMetalCopyRows(const uint8_t *src,
                      NSUInteger srcBytesPerRow,
                      uint8_t *dst,
                      NSUInteger dstBytesPerRow,
                      NSUInteger rowBytes,
                      NSUInteger height,
                      BOOL flipY);

@interface MGLRenderer ()

// === Texture upload ===
- (bool)copyTextureUploadWithDedicatedCommandBuffer:(id)sourceBuffer
                                        sourceOffset:(NSUInteger)sourceOffset
                                   sourceBytesPerRow:(NSUInteger)sourceBytesPerRow
                                 sourceBytesPerImage:(NSUInteger)sourceBytesPerImage
                                  sourceLayerStride:(NSUInteger)sourceLayerStride
                                          layerCount:(NSUInteger)layerCount
                                           sourceSize:(MGLSizeValue)sourceSize
                                            toTexture:(id)texture
                                     destinationSlice:(NSUInteger)destinationSlice
                                     destinationLevel:(NSUInteger)destinationLevel
                                    destinationOrigin:(MGLOriginValue)destinationOrigin
                                               reason:(const char *)reason;
- (bool)uploadTextureSliceViaBlit:(id)texture
                          texName:(GLuint)texName
                         texTarget:(GLenum)texTarget
                            bytes:(const void *)bytes
                      bytesPerRow:(NSUInteger)bytesPerRow
                    bytesPerImage:(NSUInteger)bytesPerImage
                            width:(NSUInteger)width
                           height:(NSUInteger)height
                            depth:(NSUInteger)depth
                            level:(NSUInteger)level
                            slice:(NSUInteger)slice;
- (bool)uploadFullCPUTextureDataIntoTexture:(Texture *)tex
                                      metal:(id)texture
                                     reason:(const char *)reason;

// === Texture readback ===
/* -syncTextureBufferFromImage: is C now (log 194; mgl_texture_mip_ops.h). */
/* -prepareImageUnitSlice: is C now (log 194; mgl_texture_mip_ops.h). */
/* -flushImageUnitSlice: is C now (log 194; mgl_texture_mip_ops.h). */
- (void)mtlReadDrawable:(GLMContext)glm_ctx
             pixelBytes:(void *)pixelBytes
            bytesPerRow:(NSUInteger)bytesPerRow
          bytesPerImage:(NSUInteger)bytesPerImage
             fromRegion:(MGLRegionValue)region;
- (void)mtlReadIntegerPixels:(GLMContext)glm_ctx
                   pixelBytes:(void *)pixelBytes
                  bytesPerRow:(NSUInteger)bytesPerRow
                bytesPerImage:(NSUInteger)bytesPerImage
                   fromRegion:(MGLRegionValue)region
                       format:(GLenum)format type:(GLenum)type;
- (void)mtlReadDepthPixels:(GLMContext)glm_ctx
                 pixelBytes:(void *)pixelBytes
                bytesPerRow:(NSUInteger)bytesPerRow
              bytesPerImage:(NSUInteger)bytesPerImage
                 fromRegion:(MGLRegionValue)region;
- (void)mtlGetTexImage:(GLMContext)glm_ctx tex:(Texture *)tex
             pixelBytes:(void *)pixelBytes bytesPerRow:(NSUInteger)bytesPerRow
          bytesPerImage:(NSUInteger)bytesPerImage fromRegion:(MGLRegionValue)region
                 format:(GLenum)format type:(GLenum)type
            mipmapLevel:(NSUInteger)level slice:(NSUInteger)slice;
/* -mtlGenerateMipmaps: is C now (log 194; mgl_texture_mip_ops.h). */
/* -mtlTexSubImage: is C now (log 198; mgl_texture_upload_ops.h). */
/* -mtlTexSubImageBytes: is C now (log 198; mgl_texture_upload_ops.h). */


// === Pending FBO clear application for readback ===

// === Locked texture upload variant ===
/* -mtlTexSubImageLocked: is C now (log 198; mgl_texture_upload_ops.h). */

// === Texture mipmap diagnostics (defined in MGLRenderer.m) ===
- (void)logMTLTextureMipmapDiagnostics:(id)mtlTexture
                                   tex:(Texture *)tex
                 effectiveMipmapLevels:(GLuint)effectiveMipmapLevels;

// === Texture upload helpers (extracted from createMTLTextureFromGLTexture:,
// defined in MGLRenderer+Texture.m) ===
- (void)reUploadExistingCPUTextureData:(Texture *)tex
                                metal:(id)texture
                          pixelFormat:(uint32_t)pixelFormat
                            numFaces:(uint)num_faces
                    uploadLevelCount:(GLuint)upload_level_count
                              isArray:(BOOL)is_array
                   texture1DBackedBy2D:(BOOL)texture1DBackedBy2D
             texture1DArrayBackedBy2DArray:(BOOL)texture1DArrayBackedBy2DArray
                             texType:(uint32_t)tex_type;

- (void)fillTextureWithSafeInitialContents:(id)texture
                                         tex:(Texture *)tex
                                 pixelFormat:(uint32_t)pixelFormat;

/* -uploadDirtyCPUTextureData: is C now (log 196; mgl_texture_upload_ops.h). */

@end

#endif /* MGLRenderer_Texture_Private_h */
