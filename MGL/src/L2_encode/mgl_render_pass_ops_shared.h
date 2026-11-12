/*
 * SPDX-License-Identifier: LGPL-3.0-only
 * Shared helpers for pass_manager_ops split TUs (Phase 7).
 */
#ifndef MGL_RENDER_PASS_OPS_SHARED_H
#define MGL_RENDER_PASS_OPS_SHARED_H

#include "mgl_renderer_ports.h"
#include "glm_context.h"
#include "mgl_render_api_texture.h"
#include "mgl_buffer_slots.h"
#include "mgl_renderer_backend.h"
#include "mgl_render_pass_manager.h"

#include <stdbool.h>
#include <stdint.h>

void mglFillLodBiasPack(GLMContext ctx, Program *prog, int stage,                               float *pack /* [4 + TEXTURE_UNITS] */);
void mglFillSamplerWrapTable(GLMContext ctx, Program *prog, int stage,                                    MGLSamplerWrapSlot *slots);
/* 1 if any sampler slot used by prog/stage has GL_MIN/GL_MAX reduction. */
int mglProgramBoundSamplersNeedReduction(GLMContext ctx, Program *prog,
                                        int stage);
GLMState *mglPdState(const MGLRendererStateAreas *areas) /* C5-EXEMPT: read helper for the write side */;
GLMState *mglPdStateView(const MGLRendererStateView *view);
void *mglRenderPassDefaultDrawBufferAttachment(    MGLRendererBackendHandle *backend, GLuint drawBufferIndex,    MGLRendererBackendDefaultDrawBufferAttachmentKind kind);
bool mglRenderPassGetPersistentState(    const MGLCommandState *commandState,    MGLRenderPassState *stateOut);
void *mglRenderPassTextureFromSnapshot(    const MGLRenderPassState *state,    uint32_t attachmentKind,    size_t colorIndex);
void mglRenderPassFillUserFboEntry(    MGLRenderPassAttachmentMatchEntry *entry, void *actual, void *expected,    int gate, const MGLRenderPassAttachmentState *snapshot,    const FBOAttachment *attachment);
bool mglPdGetPersistentAttachmentState(    const MGLCommandState *commandState, uint32_t attachmentKind,    size_t colorIndex, MGLRenderPassAttachmentState *attachmentOut);
void *mglPdAttachmentTextureFor(const MGLCommandState *commandState,                                       uint32_t attachmentKind,                                       size_t colorIndex);
bool mglPdRenderTargetSizeFor(const MGLCommandState *commandState,                                     uint64_t *widthOut, uint64_t *heightOut);
void mglPdSetPersistentAttachment(const MGLCommandState *commandState,                                         uint32_t attachmentKind,                                         size_t colorIndex, void *texture,                                         uint64_t level, uint64_t slice,                                         uint64_t depthPlane, int layered);
void mglPdSetPersistentDimensions(const MGLCommandState *commandState,                                         uint64_t width, uint64_t height);
void *mglPdDepthTextureFor(const MGLCommandState *commandState);
void *mglPdStencilTextureFor(const MGLCommandState *commandState);
bool mglPdActionsFor(const MGLCommandState *commandState,                            uint32_t attachmentKind, size_t colorIndex,                            uint32_t *loadActionOut, uint32_t *storeActionOut,                            uint64_t *storeActionOptionsOut);
void mglPdSetPersistentActions(const MGLCommandState *commandState,                                      uint32_t attachmentKind,                                      size_t colorIndex, uint32_t loadAction,                                      uint32_t storeAction);
void mglPdSetPersistentStoreAction(const MGLCommandState *commandState,                                          uint32_t attachmentKind,                                          size_t colorIndex,                                          uint32_t storeAction);
void *mglPdFallbackRenderTarget(GLMContext ctx);
void *mglPdCreateTexture(    const MGLRenderTextureDescriptorState *descriptor);
void *mglPdFallbackRenderTargetForSize(GLMContext ctx, uint64_t width,                                              uint64_t height,                                              uint64_t layerCount,                                              uint64_t sampleCount);
void *mglPdColorTextureFor(const MGLCommandState *commandState,                                 size_t colorIndex);
int mglPdNewCommandBufferTryBody(void *renderer, void *rawCtx);
int mglPdEventWaitTryBody(void *renderer, void *rawCtx);
bool mglPdClearValuesFor(const MGLCommandState *commandState,                                uint32_t attachmentKind, size_t colorIndex,                                double *clearColorOut, double *clearDepthOut,                                uint32_t *clearStencilOut);
int mglPdCreateRenderEncoderTryBody(void *renderer, void *rawCtx);
int mglPdCommitCommandBufferTryBody(void *renderer, void *rawCtx);


MGLRenderTextureInfo mglPdTextureInfo(void *texture);
uint32_t mglPdStubFSValueClass(uint32_t fmt);
uint32_t mtlPixelFormatForGLTex(Texture *gl_tex);
void *mglRenderPassDiscardStubFragmentFunction(uint32_t valueClass);
void *mglPdColorTextureFor(const MGLCommandState *commandState, size_t colorIndex);

uint64_t mglPdRenderTargetHeightFor(const MGLCommandState *commandState);

MGLRenderPassIdentityState mglRenderPassIdentitySnapshot(const MGLCommandState *commandState);
const MGLRenderPassAttachmentState *mglRenderPassAttachmentStateFromSnapshot(
    const MGLRenderPassState *state, uint32_t attachmentKind, size_t colorIndex);
uint32_t mglPdLoadActionFor(const MGLCommandState *commandState,
                            uint32_t attachmentKind, size_t colorIndex,
                            uint32_t fallback);
uint32_t mglPdStoreActionFor(const MGLCommandState *commandState,
                             uint32_t attachmentKind, size_t colorIndex,
                             uint32_t fallback);
extern const int kMglPdVerboseFrameLoopLogs;
extern GLuint mglRendererSafeFramebufferName(GLMContext ctx);

#endif
