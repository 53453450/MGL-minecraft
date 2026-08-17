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
 * Imports MGLRenderer.h for the MGLRenderer interface;
 * the category file itself imports MGLRenderer_Private.h for ivar access and shared types.
 */

#ifndef MGLRenderer_RenderPass_Private_h
#define MGLRenderer_RenderPass_Private_h

#import "MGLRenderer.h"

/* === Diagnostic constants — used by MGLRenderer.m and RenderPass/Query === */
static const BOOL kMGLDisableSharedEventSync = YES;
static const BOOL kMGLVerboseFrameLoopLogs = NO;
static const BOOL kMGLVerbosePipelineLogs = NO;

/* MSL identifier constant. */
static const char *kMGLFragCoordParamsMSLName = "_mglFragCoordParams";
static const char *kMGLLodBiasMSLName = "_mglLodBias";

/* === C functions defined in MGLRenderer.m, used by MGLRenderer+RenderPass.m === */

/* Render-pass lifecycle / pipeline helpers. */
void mglLogRenderPassLifecycle(const char *tag,
                               uint64_t call,
                               GLMContext ctx,
                               void *commandBufferOwner,
                               void *renderEncoderOwner,
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
                         void *commandBufferOwner,
                         void *renderEncoderOwner,
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
- (bool)newRenderEncoderWithReason:(MGLEncoderCreateReason)reason;
- (bool)newRenderEncoderLockedWithReason:(MGLEncoderCreateReason)reason;
- (bool)newRenderEncoder; /* OTHER — prefer WithReason: */
- (bool)newRenderEncoderLocked; /* OTHER — prefer WithReason: */
- (void)endRenderEncoding;
- (void)endRenderEncodingLocked;
- (void)endRenderPassIfFramebufferChangedForNonDraw:(uint64_t)processCall;
- (bool)currentRenderPassMatchesCurrentFramebuffer;
- (bool)bindFramebufferAttachmentTextures;
- (bool)bindBufferSizeConstantsForRenderEncoder;
- (bool)bindFramebufferTexture:(FBOAttachment *)fbo_attachment isDrawBuffer:(bool)isDrawBuffer;

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

// === Thread Safety: *Locked variants ===
- (bool)bindMTLProgram:(Program *)ptr;
- (bool)bindMTLProgramLocked:(Program *)ptr;
- (bool)newCommandBufferLocked;
- (bool)ensureWritableCommandBufferLocked:(const char *)reason;
- (void)flushCommandBufferLocked:(bool)finish;
- (bool)processGLStateLocked:(bool)draw_command;

// === Public wrapper methods (non-locking; call the *Locked variants) ===
- (bool)ensureWritableCommandBuffer:(const char *)reason;
- (bool)newCommandBuffer;
- (bool)processGLState:(bool)draw_command;
- (void)flushCommandBuffer:(bool)finish;

// === Methods defined in MGLRenderer.m, called from MGLRenderer+RenderPass.m ===
// mapBuffersToMTL, updateDirtyBaseBufferList:, checkForDirtyBufferData: are
// now declared in MGLRenderer+Buffer_Private.h (implemented in +Buffer.m).
- (id<MTLTexture>)createMTLTextureFromGLTexture:(Texture *)tex;
- (id<MTLTexture>)createFallbackMTLTexture:(Texture *)tex;
- (id<MTLSamplerState>)createMTLSamplerForTexParam:(TextureParameter *)tex_param target:(GLuint)target;
- (MTLStencilOperation)mtlStencilOpForGLOp:(GLenum)op;
- (bool)checkDrawBufferSize:(GLuint)index;
- (id)newDrawBuffer:(MTLPixelFormat)pixelFormat isDepthStencil:(bool)depthStencil;
- (id)newDrawBufferWithCustomSize:(MTLPixelFormat)pixelFormat
                     isDepthStencil:(bool)depthStencil
                        customSize:(CGSize)size;
- (MTLBlendFactor)blendFactorFromGL:(GLenum)gl_blend;
- (MTLBlendOperation)blendOperationFromGL:(GLenum)gl_blend_op;

// Thread Safety: *Locked variants defined in MGLRenderer.m
- (void)mtlDeleteMTLObjLocked:(GLMContext)glm_ctx buffer:(void *)obj;

// Locked variants defined in MGLRenderer.m (called from category files)
- (void)mtlSwapBuffersLocked:(GLMContext)glm_ctx;
- (void)mtlBufferSubDataLocked:(GLMContext)glm_ctx buf:(Buffer *)buf offset:(size_t)offset size:(size_t)size ptr:(const void *)ptr;
- (void *)mtlMapUnmapBufferLocked:(GLMContext)glm_ctx buf:(Buffer *)buf offset:(size_t)offset size:(size_t)size access:(GLenum)access map:(bool)map;
- (void)mtlFlushMappedBufferRangeLocked:(GLMContext)glm_ctx buf:(Buffer *)buf offset:(GLintptr)offset length:(GLsizeiptr)length;

// === Other methods defined in MGLRenderer.m, called from category files ===
- (BOOL)mglEnsureLayerDrawableSizeAtLeastWidth:(NSUInteger)requiredWidth
                                        height:(NSUInteger)requiredHeight
                                        reason:(const char *)reason;
- (NSUInteger)bytesPerPixelForFormat:(GLenum)internalformat;
- (CGSize)mglSyncLayerDrawableSizeFromView:(const char *)reason;

@end

#endif /* MGLRenderer_RenderPass_Private_h */
