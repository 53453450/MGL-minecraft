/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef MGL_RENDER_API_LIFECYCLE_H
#define MGL_RENDER_API_LIFECYCLE_H

/* Declarations for the lifecycle slice of the renderer facade.
 * Value layouts live in mgl_render.h. Standalone include pulls
 * mgl_render_fwd.h (incomplete types) instead of the full facade. */

#ifndef MGL_RENDER_H
#include "mgl_render_fwd.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

const char *mglRenderLoadActionName(uint32_t action);

/* Initializes the renderer. objc_device is an existing id<MTLDevice>; the C++
 * side retains it without transferring ownership. Returns 0 on success. */
int mglRenderInit(void* objc_device);

/* Load the AIR entry point named "main" with the renderer-owned device.
 * Returned library/function objects are +1 retained for the caller. */
int mglRenderLoadAIRMainFunction(const unsigned char *bytes,
                                    size_t size,
                                    void **library_out,
                                    void **function_out,
                                    char *err,
                                    size_t errcap);

void mglRenderReleaseSync(GLMContext glm_ctx, Sync *sync);

void mglRenderReleaseMetalObject(void *object);

/* RGB → RGBA channel expansion (RGBA16/RGBA32 family
 * backed by RGBA variants) — the table + verification moved from
 * mgl_texture_compat.m; malloc'd result, NULL on bad args / unknown format. */
uint8_t *mglRenderCreateChannelExpandedUpload(uint32_t internal_format,
                                                 uint32_t pixel_format,
                                                 const void *src_data,
                                                 size_t width,
                                                 size_t height,
                                                 size_t src_bytes_per_row,
                                                 size_t *out_bytes_per_row,
                                                 size_t *out_bytes_per_image);

uint8_t *mglRenderCreateRGBA8ExpandedUpload(const void *src_data,
                                               size_t width,
                                               size_t height,
                                               size_t src_bytes_per_row,
                                               uint32_t internal_format,
                                               size_t *out_bytes_per_row,
                                               size_t *out_bytes_per_image);

uint8_t *mglRenderCreateSingleChannelSwizzledUpload(
    uint32_t internal_format,
    uint32_t swizzle_r, uint32_t swizzle_g,
    uint32_t swizzle_b, uint32_t swizzle_a,
    const void *src_data, size_t width, size_t height,
    size_t src_bytes_per_row,
    size_t *out_bytes_per_row, size_t *out_bytes_per_image);

uint8_t *mglRenderCreateStencilSwizzledUpload(
    uint32_t internal_format,
    uint32_t swizzle_r, uint32_t swizzle_g,
    uint32_t swizzle_b, uint32_t swizzle_a,
    const void *src_data, size_t width, size_t height,
    size_t src_bytes_per_row,
    size_t *out_bytes_per_row, size_t *out_bytes_per_image);

int mglRenderCreateDepthStencilState(void *depth_stencil_descriptor,
                                        void **depth_stencil_state_out);

int mglRenderCreateDepthStencilStateFromState(
    const MGLRenderDepthStencilDescriptorState *descriptor,
    void **depth_stencil_state_out);

int mglRenderCreateEvent(void **event_out);

int mglRenderCreateFunction(void *library,
                               const char *name,
                               void *function_constant_values,
                               void **function_out,
                               char *err,
                               size_t errcap);

int mglRenderCreateBinaryArchive(void *binary_archive_descriptor,
                                    const char *label,
                                    void **binary_archive_out,
                                    char *err,
                                    size_t errcap);

/* Resolve entry functions from a precompiled aux shader asset for descriptor
 * paths that keep ObjC descriptor assembly (e.g. the safe fallback branch).
 * vertex_out is always a +1 MTL::Function; fragment_out is +1 when
 * fragment_entry is non-NULL. The underlying library is cached by the C++
 * renderer and released at shutdown. */
int mglRenderCreateAuxFunctions(
    const unsigned char *bytes,
    size_t size,
    uint64_t asset_hash,
    const char *vertex_entry,
    const char *fragment_entry,
    void **vertex_out,
    void **fragment_out,
    char *err,
    size_t errcap);

/* Per-command-buffer MDI argument arena. The opaque owner keeps the sole
 * persistent +1 reference; returned buffers are borrowed migration views. */
int mglRenderCreateMDIScratchOwner(MGLMDIScratchOwner **owner_out);

void mglRenderResetMDIScratchOwner(MGLMDIScratchOwner *owner);

void mglRenderDestroyMDIScratchOwner(MGLMDIScratchOwner **owner);

/* pending shared-event slot inside the C++ owner.
 * `int` in these decls is GLsizei (GL signed 32-bit) — the C ABI matches. */
int mglRenderCreatePendingEventOwner(MGLPendingEventOwner **owner_out);

void mglRenderDestroyPendingEventOwner(MGLPendingEventOwner **owner_handle);

#ifdef __cplusplus
}
#endif

#endif
