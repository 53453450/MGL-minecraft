/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef MGL_RENDER_API_READBACK_H
#define MGL_RENDER_API_READBACK_H

/* Declarations for the readback slice of the renderer facade.
 * Value layouts are defined here; types of other subsystems that are only
 * passed by pointer are incomplete (mgl_render_fwd.h). */

#include "mgl_render_fwd.h"

#ifdef __cplusplus
extern "C" {
#endif

/* C1: CopyRows + CopyDepthTextureBytesToFloat -> mgl_readback_policy.h */

/* copy GL BGRA8 rows into a BGRA8-compatible Metal pixel
 * format (RGBA8Unorm / BGRA8Unorm / RGB9E5Float / RGB10A2Unorm /
 * BGR10A2Unorm) with optional Y-flip.  Pure CPU data transform shared by
 * both gates — mirrors the ObjC
 * mglMetalCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes exactly (1 on
 * success, 0 on bad args / unsupported format). */
int mglRenderCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, int flip_y);

/* RGB10A2Unorm texture bytes -> GL format/type,
 * bypassing the lossy BGRA8 UNORM intermediate.  Mirrors the ObjC
 * sourceIsRGB10A2Direct path (1 on success, 0 on bad args / unsupported). */
int mglRenderCopyRGB10A2TextureBytesToGL(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y);

typedef struct MGLRenderReadTextureRegionClip_t {
    int32_t copy_w;
    int32_t copy_h;
    int32_t dst_x;
    int32_t dst_y;
    int32_t metal_src_x;
    int32_t metal_src_y;
    int empty;   /* copyW <= 0 || copyH <= 0 (nothing to copy) */
} MGLRenderReadTextureRegionClip;


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
