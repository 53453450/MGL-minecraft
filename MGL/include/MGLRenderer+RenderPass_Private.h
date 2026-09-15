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
                               void *renderPassStateOwner,
                               void *drawable,
                               Framebuffer *renderPassFramebuffer,
                               GLuint renderPassFramebufferName,
                               GLenum renderPassDrawBuffer,
                               GLsizei renderPassDrawBufferCount);
GLuint mglCurrentRenderProgramKey(GLMContext ctx);
void mglWriteProgramMSLDump(Program *program, const char *reason);
GLuint mglRendererSafeFramebufferName(GLMContext ctx);
id mglApplySRGBStateToRenderTarget(id texture, GLMContext ctx);
Program *mglResolveProgramFromState(GLMContext ctx);
BOOL mglRendererPointerInHashTable(HashTable *table, const void *ptr);

Program *mglResolveProgramForStageFromState(GLMContext ctx, int stage);
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
                         void *renderPassStateOwner,
                         id drawable);
Framebuffer *mglRendererGetValidatedFramebuffer(GLMContext ctx, const char *where);

/* GL type/size → Metal vertex format — defined in MGLRenderer.m. */
uint32_t glTypeSizeToMtlType(GLuint type, GLuint size, bool normalized);

/* GL texture → Metal pixel format — defined in pixel_utils.c. */
uint32_t mtlPixelFormatForGLTex(Texture *gl_tex);

/* Pipeline helper — defined in MGLRenderer.m, used by RenderPass.m and Blit.m. */

@interface MGLRenderer ()

// === Render pass state sync ===
- (bool)syncRenderPassStateForContext:(GLMContext)glm_ctx;
/* Defined in MGLRenderer+RenderPass.m; the shell port forwards to it (log 167). */
- (void)updateCurrentRenderEncoder;
- (bool)rotateRenderEncoderForCurrentFramebufferLocked;
/* -syncPipelineStateWithDeferredBufferMap: is C now (log 189):
 * mglRenderPassSyncPipelineState (mgl_pso_build_ops.h). */
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
- (void)endRenderPassIfFramebufferChangedForNonDraw:(uint64_t)processCall;
- (bool)currentRenderPassMatchesCurrentFramebuffer;
/* these are the C functions in mgl_attachment_binding.h now */
/* these are the C functions in mgl_attachment_binding.h now */

// === Framebuffer attachment helpers ===
- (Texture *)framebufferAttachmentTexture:(FBOAttachment *)fbo_attachment;
- (BOOL)currentRenderPassUsesTexture:(id)texture;
- (bool)restoreRenderEncoderAfterTextureUploadForDraw:(const char *)reason;
- (BOOL)synchronizeRenderPassForTextureReadback:(id)texture
                                          reason:(const char *)reason;

// === Thread Safety: *Locked variants ===
- (bool)bindMTLProgram:(Program *)ptr;
- (bool)bindMTLProgramLocked:(Program *)ptr;
- (bool)newCommandBufferLocked;
- (bool)processGLStateLocked:(bool)draw_command;

// === Public wrapper methods (non-locking; call the *Locked variants) ===
- (bool)ensureWritableCommandBuffer:(const char *)reason;
- (bool)newCommandBuffer;
- (bool)processGLState:(bool)draw_command;
- (void)flushCommandBuffer:(bool)finish;

// === Methods defined in MGLRenderer.m, called from MGLRenderer+RenderPass.m ===
// mapBuffersToMTL, updateDirtyBaseBufferList:, checkForDirtyBufferData: are
// now the C entries of mgl_buffer_map.h (was MGLRenderer+Buffer.m).
- (id)createMTLTextureFromGLTexture:(Texture *)tex;
- (id)createFallbackMTLTexture:(Texture *)tex;
/* createMTLSamplerForTexParam:target: is now
 * mglTextureCreateSamplerForTexParam (mgl_texture_sampler.h). */
- (bool)checkDrawBufferSize:(GLuint)index;
- (id)newDrawBuffer:(uint32_t)pixelFormat isDepthStencil:(bool)depthStencil;
- (id)newDrawBufferWithCustomSize:(uint32_t)pixelFormat
                     isDepthStencil:(bool)depthStencil
                        customSize:(CGSize)size;

// Thread Safety: *Locked variants defined in MGLRenderer.m

// Locked variants defined in MGLRenderer.m (called from category files)

// === Other methods defined in MGLRenderer.m, called from category files ===
- (BOOL)mglEnsureLayerDrawableSizeAtLeastWidth:(NSUInteger)requiredWidth
                                        height:(NSUInteger)requiredHeight
                                        reason:(const char *)reason;
/* bytesPerPixelForFormat: is now mglTextureBytesPerPixelForFormat
 * (mgl_pixel_format.h). */
- (CGSize)mglSyncLayerDrawableSizeFromView:(const char *)reason;

@end

#endif /* MGLRenderer_RenderPass_Private_h */
