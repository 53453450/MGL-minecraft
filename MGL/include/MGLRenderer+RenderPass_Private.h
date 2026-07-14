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
 * MGLRenderer+RenderPass_Private.h
 * MGL
 *
 * Private method declarations, C helpers, and constants for the RenderPass
 * category (MGLRenderer+RenderPass.m).  Also hosts method declarations
 * defined in MGLRenderer.m but called from multiple category files.
 * Imports MGLRenderer_Private.h for ivar access and shared types.
 */

#ifndef MGLRenderer_RenderPass_Private_h
#define MGLRenderer_RenderPass_Private_h

#import "MGLRenderer_Private.h"

/* === Diagnostic constants — used by MGLRenderer.m and RenderPass/Query === */
static const BOOL kMGLDisableSharedEventSync = YES;
static const BOOL kMGLVerboseFrameLoopLogs = NO;
static const BOOL kMGLVerbosePipelineLogs = NO;

/* MSL identifier constant. */
static const char *kMGLFragCoordParamsMSLName = "_mglFragCoordParams";

/* === C functions defined in MGLRenderer.m, used by MGLRenderer+RenderPass.m === */

/* Render-pass lifecycle / pipeline helpers. */
void mglLogRenderPassLifecycle(const char *tag,
                               uint64_t call,
                               GLMContext ctx,
                               id<MTLCommandBuffer> commandBuffer,
                               id<MTLRenderCommandEncoder> renderEncoder,
                               MTLRenderPassDescriptor *renderPassDescriptor,
                               id<CAMetalDrawable> drawable,
                               Framebuffer *renderPassFramebuffer,
                               GLuint renderPassFramebufferName,
                               GLenum renderPassDrawBuffer,
                               GLsizei renderPassDrawBufferCount);
NSRange mglRendererFindMSLEntryParameterClose(NSString *msl, const char *entryPoint);
GLuint mglCurrentRenderProgramKey(GLMContext ctx);
void mglWriteProgramMSLDump(Program *program, NSString *reason);
GLuint mglRendererSafeFramebufferName(GLMContext ctx);
id<MTLTexture> mglApplySRGBStateToRenderTarget(id<MTLTexture> texture, GLMContext ctx);
Program *mglResolveProgramFromState(GLMContext ctx);
BOOL mglRendererPointerInHashTable(HashTable *table, const void *ptr);

Program *mglResolveProgramForStageFromState(GLMContext ctx, int stage);
void mglNormalizePipelineDepthStencilFormats(MTLRenderPipelineDescriptor *desc,
                                             const char *label);
VertexArray *mglRendererGetValidatedVAO(GLMContext ctx, const char *where);

/* Render-pass logging / validation helpers. */
void mglLogLoopHeartbeat(const char *tag,
                         uint64_t callCount,
                         double nowSeconds,
                         double *lastCallSeconds,
                         uint64_t *lastCallCount,
                         double warnGapSeconds);
void mglLogStateSnapshot(const char *tag,
                         GLMContext ctx,
                         id<MTLCommandBuffer> commandBuffer,
                         id<MTLRenderCommandEncoder> renderEncoder,
                         MTLRenderPassDescriptor *renderPassDescriptor,
                         id<CAMetalDrawable> drawable);
Framebuffer *mglRendererGetValidatedFramebuffer(GLMContext ctx, const char *where);

/* GL type/size → Metal vertex format — defined in MGLRenderer.m. */
MTLVertexFormat glTypeSizeToMtlType(GLuint type, GLuint size, bool normalized);

/* GL texture → Metal pixel format — defined in pixel_utils.c. */
MTLPixelFormat mtlPixelFormatForGLTex(Texture *gl_tex);

/* Pipeline helper — defined in MGLRenderer.m, used by RenderPass.m and Blit.m. */
void mglEnableIndirectCommandBuffersForPipeline(MTLRenderPipelineDescriptor *pipelineStateDescriptor);

@interface MGLRenderer ()

// === Render pass state sync ===
- (bool)syncRenderPassStateForContext:(GLMContext)glm_ctx;
- (bool)rotateRenderEncoderForCurrentFramebufferLocked;
- (bool)syncPipelineStateWithDeferredBufferMap:(bool)deferredBufferMapForPipelineBuild;
- (BOOL)shouldUseDontCareLoadForColorTexture:(Texture *)tex
                             firstUseThisFrame:(BOOL)firstUseThisFrame;
- (BOOL)prepareRenderPassIfFBOChanged:(MGLDrawBatch *)batch
                              context:(GLMContext)glm_ctx
                          replayError:(GLenum *)replayError;

// === Render encoder lifecycle ===
- (bool)newRenderEncoder;
- (bool)newRenderEncoderLocked;
- (void)endRenderEncoding;
- (void)endRenderEncodingLocked;
- (bool)currentRenderPassMatchesCurrentFramebuffer;
- (bool)bindFramebufferAttachmentTextures;
- (bool)bindBufferSizeConstantsForRenderEncoder;

// === Framebuffer attachment helpers ===
- (Texture *)framebufferAttachmentTexture:(FBOAttachment *)fbo_attachment;
- (BOOL)currentRenderPassUsesTexture:(id<MTLTexture>)texture;
- (void)updateGLSampledCopiesForEndedRenderPassFramebuffer:(Framebuffer *)fbo
                                                  drawCount:(GLsizei)drawCount
                                               drawBuffers:(const GLenum *)drawBuffers
                                                    reason:(const char *)reason;
- (bool)restoreRenderEncoderAfterTextureUploadForDraw:(const char *)reason;
- (BOOL)synchronizeRenderPassForTextureReadback:(id<MTLTexture>)texture
                                          reason:(const char *)reason;

// === Phase 3 Thread Safety: *Locked variants ===
- (void)bindMTLBufferLocked:(Buffer *)ptr;
- (bool)bindMTLProgramLocked:(Program *)ptr;
- (bool)newCommandBufferLocked;
- (bool)ensureWritableCommandBufferLocked:(const char *)reason;
- (bool)bindMTLTextureLocked:(Texture *)tex;
- (void)flushCommandBufferLocked:(bool)finish;
- (bool)processGLStateLocked:(bool)draw_command;

// === Public wrapper methods (non-locking; call the *Locked variants) ===
- (bool)ensureWritableCommandBuffer:(const char *)reason;
- (bool)newCommandBuffer;
- (bool)bindMTLTexture:(Texture *)tex;
- (bool)processGLState:(bool)draw_command;
- (void)flushCommandBuffer:(bool)finish;

// === Methods defined in MGLRenderer.m, called from MGLRenderer+RenderPass.m ===
- (bool)mapBuffersToMTL;
- (id<MTLTexture>)createMTLTextureFromGLTexture:(Texture *)tex;
- (id<MTLTexture>)createFallbackMTLTexture:(Texture *)tex;
- (id<MTLSamplerState>)createMTLSamplerForTexParam:(TextureParameter *)tex_param target:(GLuint)target;
- (id<MTLLibrary>)compileShader:(const char *)str;
- (id<MTLFunction>)newFunctionFromLibrary:(id<MTLLibrary>)library
                                entryName:(NSString *)entryName
                                   source:(const char *)source
                                    label:(NSString *)label;
- (MTLStencilOperation)mtlStencilOpForGLOp:(GLenum)op;
- (bool)checkDrawBufferSize:(GLuint)index;
- (id)newDrawBuffer:(MTLPixelFormat)pixelFormat isDepthStencil:(bool)depthStencil;
- (id)newDrawBufferWithCustomSize:(MTLPixelFormat)pixelFormat
                     isDepthStencil:(bool)depthStencil
                        customSize:(CGSize)size;
- (int)getProgramBindingCount:(int)stage type:(int)type;
- (MTLBlendFactor)blendFactorFromGL:(GLenum)gl_blend;
- (MTLBlendOperation)blendOperationFromGL:(GLenum)gl_blend_op;
- (bool)updateDirtyBaseBufferList:(BufferMapList *)buffer_map_list;
- (bool)checkForDirtyBufferData:(BufferMapList *)buffer_map_list;

// === Private method declarations (defined in MGLRenderer.m, called from categories) ===
- (void)initializeMTL4CompilerIfAvailable;
- (id<MTLLibrary>)newMetalLibraryWithSource:(NSString *)source
                                    options:(MTLCompileOptions *)options
                                      label:(NSString *)label
                                      error:(NSError **)error;
- (BOOL)shouldSkipGPUOperations;
- (void)recordGPUError;
- (void)recordGPUSuccess;
- (NSUInteger)getOptimalAlignmentForPixelFormat:(MTLPixelFormat)format;
- (void)commitCommandBufferWithAGXRecovery:(id<MTLCommandBuffer>)commandBuffer;
- (void)resetMetalState;
- (BOOL)validateMetalObjects;
- (void)cleanupCommandBuffer;

// Phase 3 Thread Safety: *Locked variants defined in MGLRenderer.m
- (void)mtlDeleteMTLObjLocked:(GLMContext)glm_ctx buffer:(void *)obj;

// Locked variants defined in MGLRenderer.m (called from category files)
- (void)mtlSwapBuffersLocked:(GLMContext)glm_ctx;
- (void)mtlBufferSubDataLocked:(GLMContext)glm_ctx buf:(Buffer *)buf offset:(size_t)offset size:(size_t)size ptr:(const void *)ptr;

// === Other methods defined in MGLRenderer.m, called from category files ===
- (BOOL)mglEnsureLayerDrawableSizeAtLeastWidth:(NSUInteger)requiredWidth
                                        height:(NSUInteger)requiredHeight
                                        reason:(const char *)reason;
- (NSUInteger)bytesPerPixelForFormat:(GLenum)internalformat;
- (CGSize)mglSyncLayerDrawableSizeFromView:(const char *)reason;

/* Phase 2 #6: Binary Archive for PSO compile acceleration.
 * loadBinaryArchive loads from disk (or creates empty if not found);
 * saveBinaryArchive serializes back to disk on dealloc.
 * applyBinaryArchiveToDescriptor: attaches the archive to a pipeline
 * descriptor so PSO compile can reuse cached shader binaries. */
- (void)loadBinaryArchive;
- (void)saveBinaryArchive;
- (void)applyBinaryArchiveToDescriptor:(MTLRenderPipelineDescriptor *)descriptor;
- (void)addPipelineToBinaryArchive:(MTLRenderPipelineDescriptor *)descriptor;

@end

#endif /* MGLRenderer_RenderPass_Private_h */
