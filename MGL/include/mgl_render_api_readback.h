/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef MGL_RENDER_API_READBACK_H
#define MGL_RENDER_API_READBACK_H

/* Declarations for the readback slice of the renderer facade.
 * Value layouts live in mgl_render.h. Standalone include pulls
 * mgl_render_fwd.h (incomplete types) instead of the full facade. */

#ifndef MGL_RENDER_H
#include "mgl_render_fwd.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* readback bytes-per-pixel table (MGLPixelFormat ABI value
 * -> bytes).  Pure CPU table shared by both gates — mirrors the ObjC
 * mglMetalReadbackBytesPerPixel exactly (default 4 bytes for unlisted
 * formats).  The C ABI carries the pixel format as uint32_t (Apple stable
 * enum), matching mglRenderTextureDataKindForPixelFormat. */
uint32_t mglRenderReadbackBytesPerPixel(uint32_t pixel_format);

/* readback pixel-format classification (MGLPixelFormat ABI
 * value -> boolean).  Pure CPU tables shared by both gates — mirror the ObjC
 * mglMetalReadbackFormatIsBGRA8Compatible / mglMetalPixelFormatIsIntegerColor /
 * mglMetalPixelFormatIsSignedIntegerColor exactly.  Returns 1/0. */
int mglRenderReadbackFormatIsBGRA8Compatible(uint32_t pixel_format);

/* accepted GL pixel types for
 * mglMetalCopyBGRA8CompatibleTextureBytesToGL.  Returns 1/0. */
int mglRenderReadbackGLTypeAccepted(uint32_t type);

int mglRenderReadbackTypeIsCore(uint32_t type);

int mglRenderReadbackTypeAllowsRGB10A2(uint32_t type);

int mglRenderReadbackTypeAllowsRG11B10(uint32_t type);

int mglRenderReadbackTypeAllows16or32(uint32_t type);

int mglRenderReadbackTypeIsWideScalar(uint32_t type);

int mglRenderReadbackTypeIsPacked(uint32_t type);

int mglRenderReadbackPixelFormatIsSnorm8(uint32_t pixel_format);

int mglRenderReadbackPixelFormatIsRGB10A2(uint32_t pixel_format);

int mglRenderReadbackPixelFormatIsRG11B10(uint32_t pixel_format);

int mglRenderReadbackPixelFormatIs16or32(uint32_t pixel_format);

int mglRenderReadbackPixelFormatIsRGBA8(uint32_t pixel_format);

int mglRenderReadbackPixelFormatIsBGRA8(uint32_t pixel_format);

uint32_t mglRenderReadbackBGRA8CarrierFormat(void);

int mglRenderClearMaskHasColor(uint32_t mask);

int mglRenderClearMaskHasDepth(uint32_t mask);

int mglRenderClearMaskHasStencil(uint32_t mask);

int mglRenderClearMaskHasDepthStencil(uint32_t mask);

uint32_t mglRenderClearMaskDepthStencilBits(uint32_t mask);

uint32_t mglRenderClearMaskClearColor(uint32_t mask);

uint32_t mglRenderClearMaskClearDepth(uint32_t mask);

uint32_t mglRenderClearMaskClearStencil(uint32_t mask);

int mglRenderClearMaskHasAny(uint32_t mask);

void mglRenderClearEmptyBufferDirty(Buffer *buf);

void mglRenderClearCPUWriteRange(Buffer *buf);

int mglRenderBindingClearVertexBuffer(MGLBindingState *binding_state, uint32_t index);

int mglRenderBindingClearFragmentBuffer(MGLBindingState *binding_state, uint32_t index);

int mglRenderBindingClearFragmentTexture(MGLBindingState *binding_state, uint32_t index);

/* Kept separate from RecordSuccess to preserve the legacy two-lock completion
 * sequence. Returns 1 when recovery mode was cleared, 0 when already clear. */
int mglRenderCommandRecoveryClearMode(MGLCommandBufferRecoveryOwner *owner);

void mglRenderClearFboMatchCache(MGLRenderPassIdentityOwner *owner);

void mglRenderPendingEventClear(MGLPendingEventOwner *owner_handle);

void mglRenderCommandBufferOwnerClearSyncs(MGLCommandBufferOwner *owner_handle);

int mglRenderEncodeColorClear(void *command_buffer,
                                 void *texture,
                                 uint64_t level,
                                 uint64_t slice,
                                 uint64_t depth_plane,
                                 double red,
                                 double green,
                                 double blue,
                                 double alpha);

/* Owner-aware variant used by renderer clear paths. The current command
 * buffer remains inside CommandBufferOwner and is never borrowed through the
 * C ABI. */
int mglRenderEncodeColorClearForCommandBufferOwner(MGLCommandBufferOwner *command_buffer_owner, void *texture, uint64_t level, uint64_t slice, uint64_t depth_plane, double red, double green, double blue, double alpha);

int mglRenderEncodeDepthClear(void *command_buffer,
                                 void *texture,
                                 uint64_t level,
                                 uint64_t slice,
                                 uint64_t depth_plane,
                                 double clear_depth);

int mglRenderEncodeDepthClearForCommandBufferOwner(MGLCommandBufferOwner *command_buffer_owner, void *texture, uint64_t level, uint64_t slice, uint64_t depth_plane, double clear_depth);

#ifdef __cplusplus
}
#endif

#endif
