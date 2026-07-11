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

#import "MGLRenderer_Private.h"

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
- (bool)copyTextureUploadWithDedicatedCommandBuffer:(id<MTLBuffer>)sourceBuffer
                                        sourceOffset:(NSUInteger)sourceOffset
                                   sourceBytesPerRow:(NSUInteger)sourceBytesPerRow
                                 sourceBytesPerImage:(NSUInteger)sourceBytesPerImage
                                           sourceSize:(MTLSize)sourceSize
                                            toTexture:(id<MTLTexture>)texture
                                     destinationSlice:(NSUInteger)destinationSlice
                                     destinationLevel:(NSUInteger)destinationLevel
                                    destinationOrigin:(MTLOrigin)destinationOrigin
                                               reason:(const char *)reason;
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
                            slice:(NSUInteger)slice;
- (bool)uploadFullCPUTextureDataIntoTexture:(Texture *)tex
                                      metal:(id<MTLTexture>)texture
                                     reason:(const char *)reason;

// === Texture readback ===
- (void)mtlReadDrawable:(GLMContext)glm_ctx
             pixelBytes:(void *)pixelBytes
            bytesPerRow:(NSUInteger)bytesPerRow
          bytesPerImage:(NSUInteger)bytesPerImage
             fromRegion:(MTLRegion)region;

// === Pending FBO clear application for readback ===
- (void)mglApplyPendingFBODepthClearForReadback:(Framebuffer *)fbo
                                     attachment:(FBOAttachment *)attachment
                                    textureObj:(Texture *)textureObj
                                     mtlTexture:(id<MTLTexture>)texture;
- (void)mglApplyPendingFBOColorClearForReadback:(Framebuffer *)fbo
                                     attachment:(FBOAttachment *)attachment
                                    textureObj:(Texture *)textureObj
                                     mtlTexture:(id<MTLTexture>)texture
                                  attachmentEnum:(GLenum)attachmentEnum;

// === Locked texture upload variant ===
- (void)mtlTexSubImageLocked:(GLMContext)glm_ctx tex:(Texture *)tex buf:(Buffer *)buf src_offset:(size_t)src_offset src_pitch:(size_t)src_pitch src_image_size:(size_t)src_image_size src_size:(size_t)src_size slice:(GLuint)slice level:(GLuint)level width:(size_t)width height:(size_t)height depth:(size_t)depth xoffset:(size_t)xoffset yoffset:(size_t)yoffset zoffset:(size_t)zoffset;

// === Texture mipmap diagnostics (defined in MGLRenderer.m) ===
- (void)logMTLTextureMipmapDiagnostics:(id<MTLTexture>)mtlTexture
                                   tex:(Texture *)tex
                 effectiveMipmapLevels:(GLuint)effectiveMipmapLevels;

// === Texture upload helpers (extracted from createMTLTextureFromGLTexture:,
// defined in MGLRenderer+Texture.m) ===
- (void)reUploadExistingCPUTextureData:(Texture *)tex
                                metal:(id<MTLTexture>)texture
                          pixelFormat:(MTLPixelFormat)pixelFormat
                            numFaces:(uint)num_faces
                    uploadLevelCount:(GLuint)upload_level_count
                              isArray:(BOOL)is_array
                   texture1DBackedBy2D:(BOOL)texture1DBackedBy2D
             texture1DArrayBackedBy2DArray:(BOOL)texture1DArrayBackedBy2DArray
                             texType:(MTLTextureType)tex_type;

- (void)fillTextureWithSafeInitialContents:(id<MTLTexture>)texture
                                         tex:(Texture *)tex
                                 pixelFormat:(MTLPixelFormat)pixelFormat;

- (BOOL)uploadDirtyCPUTextureData:(Texture *)tex
                            metal:(id<MTLTexture>)texture
                      pixelFormat:(MTLPixelFormat)pixelFormat
                        numFaces:(uint)num_faces
                uploadLevelCount:(GLuint)upload_level_count
                         isArray:(BOOL)is_array
              texture1DBackedBy2D:(BOOL)texture1DBackedBy2D
        texture1DArrayBackedBy2DArray:(BOOL)texture1DArrayBackedBy2DArray
                         texType:(MTLTextureType)tex_type
            outAllLevelsUploaded:(BOOL *)outAllLevelsUploaded;

@end

#endif /* MGLRenderer_Texture_Private_h */
