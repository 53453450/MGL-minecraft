/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef MGL_RENDER_API_LIFECYCLE_H
#define MGL_RENDER_API_LIFECYCLE_H

/* Declarations for the lifecycle slice of the renderer facade.
 * Value layouts are defined here; types of other subsystems that are only
 * passed by pointer are incomplete (mgl_render_fwd.h). */

#include "mgl_render_fwd.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Releases renderer-owned MTL::* objects, including the retained device.
 * This operation is idempotent. */
void mglRenderShutdown(void);

/* Direct renderer entries. Objects passed here carry the +1 bridge reference
 * owned by the GL state. */
void mglRenderDeleteMTLObj(GLMContext glm_ctx, void *object);

/* Publish the borrowed runtime owner handles used by direct C++ callbacks. */
int mglRenderAttachRuntimeOwners(GLMContext glm_ctx, MGLCommandBufferOwner *command_buffer_owner, MGLRenderEncoderOwner *render_encoder_owner, MGLRenderPassStateOwner *render_pass_state_owner);

void mglRenderDetachRuntimeOwners(GLMContext glm_ctx);

const char *mglRenderGLSLTypeName(uint32_t type);

uint32_t mglRenderGLSLMatrixCols(uint32_t type);

uint32_t mglRenderGLSLMatrixRows(uint32_t type);

const char *mglRenderGLSLColumnType(uint32_t rows);

int mglRenderGLSLNeedsFlat(uint32_t type);

typedef struct MGLRenderStencilDescriptorState_t {
    uint32_t present;
    uint32_t compare_function;
    uint32_t read_mask;
    uint32_t write_mask;
    uint32_t stencil_failure_operation;
    uint32_t depth_failure_operation;
    uint32_t depth_stencil_pass_operation;
} MGLRenderStencilDescriptorState;

typedef struct MGLRenderDepthStencilDescriptorState_t {
    uint32_t depth_compare_function;
    uint32_t depth_write_enabled;
    MGLRenderStencilDescriptorState front;
    MGLRenderStencilDescriptorState back;
} MGLRenderDepthStencilDescriptorState;

/* Read an opaque ObjC depth/stencil descriptor into value-state. The descriptor
 * object is borrowed and inspected only inside the Metal-cpp implementation TU. */
int mglRenderDescribeDepthStencilDescriptor(
    const void *depth_stencil_descriptor,
    MGLRenderDepthStencilDescriptorState *state_out);

enum {
    MGL_RENDER_AUX_COMPUTE_SCALED_BLIT = 1,
    MGL_RENDER_AUX_COMPUTE_MSAA_INTEGER_RESOLVE = 2,
    MGL_RENDER_AUX_RENDER_SCALED_BLIT = 3,
    MGL_RENDER_AUX_RENDER_SCALED_DEPTH_BLIT = 4,
    MGL_RENDER_AUX_RENDER_CLEAR_RECT = 5,
    MGL_RENDER_AUX_COMPUTE_GS_XFB_SCATTER = 6,
    MGL_RENDER_AUX_COMPUTE_EMULATED_MS_RESOLVE = 7,
};

/* Lookup or create a renderer-lifetime auxiliary render PSO. Passing NULL for
 * both functions performs lookup only and returns 1 on a cache miss. The
 * descriptor contains the fixed-format blit/clear surface state; functions
 * are the actual MTLFunction objects compiled by the caller. */
int mglRenderGetOrCreateAuxRenderPipeline(
    void *vertex_function,
    void *fragment_function,
    uint32_t kind,
    uint64_t variant,
    uint32_t color_format,
    uint32_t depth_format,
    uint32_t stencil_format,
    uint32_t color_write_mask,
    int icb_enabled,
    uint32_t raster_sample_count,
    void **pipeline_out,
    char *err,
    size_t errcap);

/* Aux render PSO from the precompiled aux shader asset table
 * (see mgl_aux_assets.h). bytes/size/hash come from an embedded table row;
 * the C++ side validates size and the FNV-1a hash, loads MTL::Library from the
 * bytes, resolves the entry functions, and creates the PSO through the same
 * renderer-lifetime cache as the function-based path. vertex_entry is the
 * metallib entry name; fragment_entry may be NULL for fragment-less kinds.
 * On success *pipeline_out is a +1 MTL::RenderPipelineState reference. */
int mglRenderGetOrCreateAuxRenderPipelineFromMetallib(
    const unsigned char *bytes,
    size_t size,
    uint64_t asset_hash,
    const char *vertex_entry,
    const char *fragment_entry,
    uint32_t kind,
    uint64_t variant,
    uint32_t color_format,
    uint32_t depth_format,
    uint32_t stencil_format,
    uint32_t color_write_mask,
    int icb_enabled,
    uint32_t raster_sample_count,
    void **pipeline_out,
    char *err,
    size_t errcap);

uint64_t mglXfbAdvanceWriteOffset(uint64_t current, uint64_t written);

int mglRenderAllocateMDIScratch(MGLMDIScratchOwner *owner, uint64_t length, uint64_t alignment, void **buffer_out, uint64_t *offset_out, uint64_t *capacity_out);


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
