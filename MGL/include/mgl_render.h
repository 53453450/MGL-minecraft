/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

//------------------------------------------------------------------------------------------------
// Pure C entry points for the C++ renderer facade.
//
// The GL state layer and Objective-C shell use this header without exposing
// MTL::* types. mgl_render.cpp is the only metal-cpp implementation TU.
//------------------------------------------------------------------------------------------------
#pragma once

#ifndef MGL_RENDER_H
#define MGL_RENDER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
/* This header uses GLuint/GLenum/GLboolean in its own declarations
 * (mglCurrentRenderProgramKey, mglRestoreProgramPipelinePair, ...) but used to
 * receive them only transitively, via glm_context.h -> mgl_renderer_backend.h.
 * Breaking that cycle exposed five TUs that included this header directly and
 * therefore had no GL types at all.  Include the registry here so the facade
 * is self-contained. */
#include <GL/glcorearb.h>
#include "mgl_render_values.h"
#include "mgl_readback_policy.h"
#include "mgl_binding_policy.h"
#include "mgl_binding_stage.h"
#include "mgl_binding_texture.h"
#include "mgl_pso_format_class.h"

/* Forward decl (mgl_types_texture.h pulls in GLMContext-typed state). */
typedef struct TextureLevel_t TextureLevel;

typedef struct GLMContextRec_t *GLMContext;
typedef struct Buffer_t Buffer;
typedef struct Texture_t Texture;
typedef struct TextureParameter_t TextureParameter;
typedef struct Program_t Program;
typedef struct __GLsync Sync;

typedef struct MGLMetalAttachmentSubresource_t MGLMetalAttachmentSubresource;

/* Value-state pipeline descriptor defined in mgl_air_loader.h. Objective-C
 * constructs this state without exposing MTLRenderPipelineDescriptor. */
typedef struct MGLRenderPipelineDescriptorState
    MGLRenderPipelineDescriptorState;

/* Device capability snapshot produced by the Metal-cpp owner.  The C ABI
 * carries only integer/value state; the MTL::Device is used exclusively by
 * mgl_render.cpp while populating this record. */
typedef struct MGLRenderCapabilityState_t {
    uint32_t family;
    uint32_t is_virtualized;
    uint32_t supports8x_msaa;
    uint64_t max_sample_count;
    uint64_t max_texture_dimensions;
    uint32_t bug_3d_getbytes_slice_oob;
    uint32_t bug_3d_replace_region_nonzero_origin;
    uint32_t bug_msl_pipeline_rejection;
    uint64_t command_buffer_recovery_limit;
    uint64_t max_concurrent_command_buffers;
    uint64_t texture_alignment_bytes;
    uint32_t conservative_cpu_cache_mode;
} MGLRenderCapabilityState;

/* Opaque owners. The C++ layout lives in mgl_render_internal.h. */
typedef struct MGLBindingState MGLBindingState;
typedef struct MGLCommandBufferOwner MGLCommandBufferOwner;
typedef struct MGLCommandBufferRecoveryOwner MGLCommandBufferRecoveryOwner;
typedef struct MGLCommandQueueOwner MGLCommandQueueOwner;
typedef struct MGLCullDistanceIndexPlan MGLCullDistanceIndexPlan;
typedef struct MGLMDIScratchOwner MGLMDIScratchOwner;
typedef struct MGLPendingEventOwner MGLPendingEventOwner;
typedef struct MGLPipelineCacheOwner MGLPipelineCacheOwner;
typedef struct MGLQueryStateOwner MGLQueryStateOwner;
typedef struct MGLRenderEncoderOwner MGLRenderEncoderOwner;
typedef struct MGLRenderPassIdentityOwner MGLRenderPassIdentityOwner;
typedef struct MGLRenderPassStateOwner MGLRenderPassStateOwner;
typedef struct MGLTextureStagingOwner MGLTextureStagingOwner;

#ifdef __cplusplus
extern "C" {
#endif

const char *mglRenderStoreActionName(uint32_t action);


/* Releases renderer-owned MTL::* objects, including the retained device.
 * This operation is idempotent. */
void mglRenderShutdown(void);




/* Direct renderer entries. Objects passed here carry the +1 bridge reference
 * owned by the GL state. */
void mglRenderDeleteMTLObj(GLMContext glm_ctx, void *object);
/* --- renderer-side program / framebuffer-binding helpers ------------------
 *
 * These three are DEFINED IN C (mgl_renderer_entries.c, mgl_draw_support.c)
 * but their only declarations used to live in the ObjC private headers
 * (MGLRenderer+RenderPass_Private.h, MGLRenderer+Draw_Private.h), which C
 * translation units must not include.  C callers therefore either hand-declared
 * them locally (mglResolveProgramForStageFromState had four such local externs)
 * or relied on a duplicate declaration that this header used to carry.
 *
 * The declaration belongs here: this header is the C-facing renderer facade.
 * The duplicates in the private headers are the ones that must go, not these. */

/* Program bound to a stage by GL state: GL_CURRENT_PROGRAM when one is bound,
 * otherwise the separable pipeline's stage program. */
Program *mglResolveProgramForStageFromState(GLMContext ctx, int stage);
/* Identity of the program the current draw would use (restored/scheduled), as
 * the trace log and pipeline-cache keys see it. */
GLuint mglCurrentRenderProgramKey(GLMContext ctx);
/* Re-sync the framebuffer binding names after a hashtable swap. */
void mglRendererSyncFramebufferBindingNames(GLMContext ctx);
/* Restore the program/pipeline pair a state key names. */
/* single source: mglRestoreProgramPipelinePair (C TUs and the ObjC private headers both
 * resolve it from here; the private copies were duplicates) */
void mglRestoreProgramPipelinePair(GLMContext ctx, GLuint programName,
                                   GLuint pipelineName);
void mglRenderWaitForSync(GLMContext glm_ctx, Sync *sync);
void mglRenderFlush(GLMContext glm_ctx, bool finish);

/* Publish the borrowed runtime owner handles used by direct C++ callbacks. */
int mglRenderAttachRuntimeOwners(GLMContext glm_ctx, MGLCommandBufferOwner *command_buffer_owner, MGLRenderEncoderOwner *render_encoder_owner, MGLRenderPassStateOwner *render_pass_state_owner);
void mglRenderDetachRuntimeOwners(GLMContext glm_ctx);

enum {
    MGL_RENDER_AIR_PROGRAM_BOUND = 0,
    MGL_RENDER_AIR_PROGRAM_NOT_APPLICABLE = 1,
    MGL_RENDER_AIR_PROGRAM_ERROR = -1,
};


enum {
    MGL_RENDER_BUFFER_BOUND = 0,
    MGL_RENDER_BUFFER_NOT_APPLICABLE = 1,
    MGL_RENDER_BUFFER_ERROR = -1,
};


enum {
    MGL_RENDER_BUFFER_OPERATION_HANDLED = 0,
    MGL_RENDER_BUFFER_OPERATION_NOT_APPLICABLE = 1,
    MGL_RENDER_BUFFER_OPERATION_ERROR = -1,
};


typedef enum MGLRenderVertexConversionKind_t {
    MGL_RENDER_VERTEX_DOUBLE_TO_FLOAT = 0,
    MGL_RENDER_VERTEX_INT_TO_FLOAT = 1,
    MGL_RENDER_VERTEX_FIXED_TO_FLOAT = 2,
    MGL_RENDER_VERTEX_PACKED_1010102_TO_FLOAT = 3,
    MGL_RENDER_VERTEX_PACKED_10F11F11F_TO_FLOAT = 4,
    MGL_RENDER_VERTEX_INTEGER_TO_32 = 5,
} MGLRenderVertexConversionKind;

typedef struct MGLRenderVertexConversion_t {
    uint32_t kind;
    uint32_t component_count;
    uint32_t source_type;
    uint32_t normalized;
    uint32_t destination_signed;
    int64_t binding_offset;
    int64_t relative_offset;
    uint64_t stride;
} MGLRenderVertexConversion;




typedef struct MGLRenderBufferInfo_t {
    uint64_t length;
} MGLRenderBufferInfo;
typedef struct MGLRenderTextureInfo_t {
    uint32_t pixel_format;
    uint32_t texture_type;
    uint64_t width;
    uint64_t height;
    uint64_t depth;
    uint64_t mipmap_level_count;
    uint64_t array_length;
    uint64_t usage;
    uint32_t storage_mode;
    uint64_t sample_count;
} MGLRenderTextureInfo;
typedef struct MGLRenderTextureDescriptorState_t {
    uint32_t texture_type;
    uint32_t pixel_format;
    uint64_t width;
    uint64_t height;
    uint64_t depth;
    uint64_t mipmap_level_count;
    uint64_t sample_count;
    uint64_t array_length;
    uint64_t resource_options;
    uint64_t usage;
    uint32_t cpu_cache_mode;
    uint32_t storage_mode;
    uint32_t hazard_tracking_mode;
    uint32_t compression_type;
    uint32_t placement_sparse_page_size;
    uint32_t allow_gpu_optimized_contents;
    uint32_t swizzle_red;
    uint32_t swizzle_green;
    uint32_t swizzle_blue;
    uint32_t swizzle_alpha;
    uint32_t has_swizzle;
} MGLRenderTextureDescriptorState;

int mglRenderCreateTextureViewRange(
    void *texture,
    uint32_t pixel_format,
    uint32_t texture_type,
    uint64_t level_location,
    uint64_t level_length,
    uint64_t slice_location,
    uint64_t slice_length,
    int use_swizzle,
    uint32_t swizzle_red,
    uint32_t swizzle_green,
    uint32_t swizzle_blue,
    uint32_t swizzle_alpha,
    void **texture_view_out);
/* CPU-visible texture transfer facade. use_slice selects Metal's
 * slice/bytesPerImage overload; region values are passed explicitly so the
 * C ABI does not expose MTLRegion. */
int mglRenderTextureReplaceRegion(void *texture,
                                     uint64_t x,
                                     uint64_t y,
                                     uint64_t z,
                                     uint64_t width,
                                     uint64_t height,
                                     uint64_t depth,
                                     uint64_t level,
                                     uint64_t slice,
                                     const void *bytes,
                                     uint64_t bytes_per_row,
                                     uint64_t bytes_per_image,
                                     int use_slice);
int mglRenderTextureGetBytes(void *texture,
                                void *bytes,
                                uint64_t bytes_per_row,
                                uint64_t bytes_per_image,
                                uint64_t x,
                                uint64_t y,
                                uint64_t z,
                                uint64_t width,
                                uint64_t height,
                                uint64_t depth,
                                uint64_t level,
                                uint64_t slice,
                                int use_slice);

/* GL texture creation target + sample count -> Metal descriptor shape.
 * The value result keeps Metal enums behind uint32_t and carries the legacy
 * upload/completeness flags that must stay consistent with the chosen type.
 * GL_TEXTURE_BUFFER is handled by its dedicated buffer-texture path before
 * this helper is called. */
typedef struct MGLRenderTextureTargetPlan_t {
    uint32_t texture_type;
    uint32_t num_faces;
    uint32_t is_array;
    uint32_t texture_1d_backed_by_2d;
    uint32_t texture_1d_array_backed_by_2d_array;
} MGLRenderTextureTargetPlan;


/* GL subimage coordinates -> Metal upload subresource plan.  In
 * particular, GL_TEXTURE_1D_ARRAY stores its first layer/count in
 * yoffset/height, while the Metal 2D-array backing needs slice/arrayLength
 * with origin.y=0 and height=1.  The result is pure value state; no MTL type
 * crosses the C ABI. */
typedef struct MGLRenderTextureSubUploadPlan_t {
    uint64_t destination_base_slice;
    uint64_t destination_x;
    uint64_t destination_y;
    uint64_t destination_z;
    uint64_t copy_width;
    uint64_t copy_height;
    uint64_t copy_depth;
    uint64_t layer_count;
    uint64_t source_layer_stride;
} MGLRenderTextureSubUploadPlan;

int mglRenderTextureSubUploadPlan(
    uint32_t gl_target,
    uint32_t texture_type,
    uint64_t requested_slice,
    uint64_t xoffset,
    uint64_t yoffset,
    uint64_t zoffset,
    uint64_t width,
    uint64_t height,
    uint64_t depth,
    uint64_t source_bytes_per_row,
    uint64_t source_bytes_per_image,
    MGLRenderTextureSubUploadPlan *plan_out);



/* MGLPixelFormat ABI value -> shader-visible texture
 * data kind.  Keep the C ABI backend-neutral; the numeric results mirror
 * MGLTextureDataKind without exposing that ObjC enum here. */
#ifndef MGL_RENDER_TEXTURE_DATA_KIND_UNKNOWN
#define MGL_RENDER_TEXTURE_DATA_KIND_UNKNOWN 0u
#define MGL_RENDER_TEXTURE_DATA_KIND_FLOAT   1u
#define MGL_RENDER_TEXTURE_DATA_KIND_SINT    2u
#define MGL_RENDER_TEXTURE_DATA_KIND_UINT    3u
#define MGL_RENDER_TEXTURE_DATA_KIND_DEPTH   4u
#endif




uint32_t mglRenderSRGBPixelFormat(uint32_t pixel_format);

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






/* Selects the CPU-to-GPU upload route without touching Metal objects.
 * texture_type and storage_mode use the stable MTLTextureType and
 * MTLStorageMode ABI values.
 *   - Non-private 1D/1DArray textures use REPLACE_1D.
 *   - 3D textures affected by the AGX slice-copy issue reject private storage;
 *     other 3D textures use REPLACE_3D with tightly packed depth planes.
 *   - Other texture shapes use BLIT to preserve GPU ordering. */
#define MGL_RENDER_TEXTURE_UPLOAD_ROUTE_BLIT          0
#define MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REPLACE_1D    1
#define MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REPLACE_3D    2
#define MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REJECT        3

/* Complete value-state plan for a full texture-level/slice upload.  This
 * centralizes the layout normalization that used to live around the ObjC
 * replaceRegion/blit branches.  A REJECT route is a valid plan; malformed
 * dimensions/strides and staging allocations above 512 MiB return -1. */
typedef struct MGLRenderTextureUploadPlan_t {
    uint32_t route;
    uint32_t replace_region_dimension; /* 1, 2, or 3; 0 for blit/reject */
    uint32_t replace_use_slice;
    uint32_t requires_repack;
    uint64_t normalized_height;
    uint64_t normalized_depth;
    uint64_t upload_rows;
    uint64_t expected_bytes_per_image;
    uint64_t normalized_bytes_per_image;
    uint64_t copy_depth;
    uint64_t buffer_size;
    uint64_t destination_slice;
    uint64_t destination_level;
} MGLRenderTextureUploadPlan;

int mglRenderBuildTextureUploadPlan(
    uint32_t gl_target,
    uint32_t texture_type,
    uint32_t storage_mode,
    uint32_t pixel_format,
    int has_agx_3d_copy_bug,
    uint64_t width,
    uint64_t height,
    uint64_t depth,
    uint64_t bytes_per_row,
    uint64_t bytes_per_image,
    uint64_t destination_level,
    uint64_t destination_slice,
    MGLRenderTextureUploadPlan *plan_out);
/* C1: TextureRepackDepthPlanes -> mgl_readback_policy.h */

/* Expands legacy packed GL formats into RGBA8. Returns a malloc-owned buffer,
 * or NULL for invalid input, unsupported formats, or size overflow. */
/* stage-binding copy-back entry (C-ABI mirror of the
 * ObjC MGLStageBindingCopyBack — the ObjC side bridges the buffer refs). */
typedef struct MGLRenderCopyBackEntry_t {
    const void *temporary;        /* MTL::Buffer* */
    const void *destination;      /* MTL::Buffer* */
    const void *destination_buffer; /* GL Buffer* (CPU prefix sync) */
    uint64_t destination_offset;
    uint64_t length;
} MGLRenderCopyBackEntry;

uint32_t mglRenderCollectCopyBackEntries(const MGLRenderCopyBackEntry *slots,
                                         uint32_t slot_count,
                                         MGLRenderCopyBackEntry *out,
                                         uint32_t out_cap);


/* Synchronize the written CPU prefix of each entry's GL destination buffer
 * (guards + memmove; the Metal contents pointer is read via the
 * destination buffer).  Returns 0, or -1 with *failed_index_out set. */
int mglRenderCopyBackCPUPrefix(
    const MGLRenderCopyBackEntry *entries,
    uint32_t count,
    uint32_t *failed_index_out);

/* runtime-array-size SSBO sizing constants.  The AIR
 * backend emits code that reads uint32 byte-sizes from
 * MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX when a compute shader uses .length()
 * on an unsized SSBO array.  This fills `out_sizes[out_capacity]` from the
 * per-buffer {metal_slot, visible_size} pairs, skipping the runtime-size
 * buffer slot itself and any slot >= max_slot (the ordinary user-buffer table
 * cap, kMGLMaxMetalUserBufferCount=31).  `out_sizes` is expected to be
 * zero-initialized by the caller; only claimed slots are written.  Returns
 * 0 on success, -1 on bad args (NULL out, NULL entries with nonzero count,
 * out_capacity < max_slot). */
typedef struct MGLRenderBufferSizeEntry_t {
    uint32_t metal_slot;      /* Metal buffer argument index */
    uint64_t visible_size;    /* byte size, truncated to uint32 by the facade */
} MGLRenderBufferSizeEntry;

int mglRenderBuildRuntimeArraySizes(
    const MGLRenderBufferSizeEntry *entries,
    uint32_t entry_count,
    uint32_t runtime_buffer_index,
    uint32_t max_slot,
    uint32_t *out_sizes,
    uint32_t out_capacity);

/* C1: IntegerReadback + Y-flip/depth/GetTexImagePlan/MSAA stride -> mgl_readback_policy.h */






/* TES XFB field byte size for a GL type (FLOAT/INT/
 * UINT + vec2/3/4; 0 for unsupported).  Matches mglTESXFBFieldByteSize and
 * the packed-write stride contract in mglFixMSLTesAsComputeKernel.  Shared
 * by both gates. */
uint64_t mglRenderTESXFBFieldByteSize(uint64_t gl_type);

/* overflow-checked product (a * b) for tessellation
 * size math; matches the ObjC mglCheckedNSUIntegerProduct.  Returns 0 with
 * *result set, -1 on bad args / overflow.  Shared by both gates. */
int mglRenderCheckedProduct(uint64_t a, uint64_t b, uint64_t *result);





/* TES XFB compact vertex stride — sum of the byte
 * sizes of the transform-feedback varyings resolved by name against the
 * TES stage-output resource list (lockstep with the packed writes injected
 * by mglFixMSLTesAsComputeKernel).  0 when the stride cannot be proven
 * (no varyings / unknown field type / overflow).  Matches the ObjC
 * mglTESXFBVertexStride.  Shared by both gates. */
uint64_t mglRenderTESXFBVertexStride(const void *program);


/* native TES interface support decision — module /
 * function presence, point-mode / XFB exclusion, TRI/QUADS gen-mode gate,
 * and the MTL::Function patchType + patchControlPointCount consistency
 * checks (zero control-point count = legacy encoding, tolerated).  Shared
 * by both gates; the ObjC caller passes __bridge'd MTL::Function pointers. */
int mglRenderNativeTESInterfaceSupported(
    void *tes_function,
    uint64_t tes_metallib_bytes,
    uint32_t tes_gen_point_mode,
    uint32_t tes_xfb_varying_count,
    uint32_t tes_gen_mode,
    void *tcs_function,
    uint64_t tcs_metallib_bytes,
    uint32_t tcs_output_vertices);

/* pure viewport/scissor/framebuffer intersection
 * decision for the per-draw rasterization-empty early-out.  Returns 1 when
 * the draw cannot rasterize any pixel, 0 otherwise (a zero pass size is
 * "not empty" — the caller resolves the pass size first).  Shared by both
 * gates. */
int mglRenderRasterizationIsEmpty(
    int32_t vx,
    int32_t vy,
    int32_t vw,
    int32_t vh,
    uint32_t pass_width,
    uint32_t pass_height,
    int32_t scissor_enabled,
    int32_t sx,
    int32_t sy,
    int32_t sw,
    int32_t sh);








/* quad-array emulation — for each group of 4 array
 * vertices emit `(a,a+1,a+2,a,a+2,a+3)` (two triangles), quad_count*6
 * uint32 total.  Pure CPU; caller frees.  Returns 0 with *out_count, -1 on
 * bad args. */
int mglRenderExpandQuadArrayIndices(
    uint32_t quad_count, uint32_t **out_indices, uint64_t *out_count);








/* index-range scan ignoring primitive-restart markers
 * — computes min/max over the byte stream (BYTE/SHORT/INT width), skipping
 * the restart value.  Pure CPU; matches mglScanIndexRangeIgnoringRestart.
 * Returns 0 on success (with *out_valid = 1 if at least one non-restart
 * index was seen), -1 on bad args. */
/* Convert a restart-aware index span into the inclusive [first, first+count)
 * vertex range used by cull-distance capture. Returns 0 on success. */
/* prepared (Metal-side) byte offset for a GL element
 * buffer — GL_UNSIGNED_BYTE indices are expanded to UInt16 so the offset
 * doubles, other types pass through.  Matches mglComputePreparedIndexByteOffset.
 * Returns 0 on success, -1 on overflow / bad args. */
/* baseByteOffset + firstElement * indexStride with
 * overflow checks.  Matches mglComputeIndexByteOffset.  Returns 0 on success,
 * -1 on bad args / overflow. */
/* GL index element byte size (BYTE=1, SHORT=2, INT=4).
 * Matches mglGLIndexElementSize.  Returns 0 for unknown type. */

/* read a single GL index value from a byte buffer at
 * `element_index` (elem_width 1/2/4).  Matches mglReadGLIndexValue; returns 0
 * for NULL buffer or unknown width. */
/* GL vertex-attribute component size in bytes (1/2/4/8).
 * Matches mglVertexAttribComponentSize.  Returns 0 for unknown. */

/* total bytes for a vertex-attribute element (type x
 * size), with special handling for packed 10_10_10_2 formats.  Matches
 * mglVertexAttribElementBytes.  Returns 0 for unknown / zero size. */

enum {
    MGL_ATTRIB_SPAN_OK = 0,
    MGL_ATTRIB_SPAN_NEGATIVE_RELATIVE = -1,
    MGL_ATTRIB_SPAN_OVERFLOW = -2,
};


enum {
    MGL_ATTRIB_FETCH_OK = 0,
    MGL_ATTRIB_FETCH_BAD_FORMAT = 1,
    MGL_ATTRIB_FETCH_OVERFLOW = 2,
    MGL_ATTRIB_FETCH_OOB = 3,
};

typedef struct MGLRenderAttribFetchPlan {
    uint32_t status;
    uint64_t elem_bytes;
    uint64_t stride;
    uint64_t rel_offset;
    uint64_t byte_start;
    uint64_t byte_end;
} MGLRenderAttribFetchPlan;

uint64_t mglRenderExpectedArrayLayers(uint32_t gl_target, int32_t depth);
int mglRenderPrefer1DOverDefault2D(uint32_t expected_type, uint32_t active_target);
int mglRenderPreferMSOr1DArrayOver2DArray(uint32_t expected_type,
                                          uint32_t active_target);
int mglRenderExpectedTypeIsCube(uint32_t expected_type);
uint64_t mglRenderFallbackSampledCacheKey(uint32_t texture_type, uint32_t data_kind);
uint32_t mglRenderAGXCompatiblePixelFormat(uint32_t pixel_format, int *converted);
int mglRenderPromote1DArrayDepthStencil(uint32_t tex_type, uint32_t pixel_format);
int mglRenderPromoteMipmapped1D(uint32_t tex_type);
int mglRenderPromoteMipmapped1DArray(uint32_t tex_type);
int mglRenderCubeFaceSizeValid(uint64_t width, uint64_t height);
uint32_t mglRenderUploadLevelCount(int mipmapped, int tex_mipmapped,
                                   uint32_t effective);
int mglRenderEmulateMSAsArray(uint32_t tex_type, uint32_t samples, uint64_t depth,
                              uint32_t *out_type, uint32_t *sample_count,
                              uint64_t *array_len, uint64_t *depth_out);
int mglRenderPreferSharedStorage(int needs_cpu, int is_depth_stencil);
void mglRenderApply1DBackingToDesc(int backed_1d, int backed_1d_array,
                                   uint64_t height, uint32_t *type,
                                   uint64_t *array_len, uint32_t *height_out);
int mglRenderTraceR8RedUByte(uint32_t internalformat, uint32_t format,
                             uint32_t type);
uint32_t mglRenderCompletenessCheckFaces(uint32_t target, uint32_t num_faces);
int mglRenderIs3DReupload(uint32_t target, uint32_t depth);
uint32_t mglRenderDepthStencilPlaneViewType(uint32_t parent_type);
uint32_t mglRenderStencilViewFormat(uint32_t parent_format);
uint32_t mglRenderRepairedDefaultStencilFormat(uint32_t stencil_format);
int mglRenderPackedD32FNeeds8ByteStride(uint32_t pixel_format,
                                        uint32_t row_bytes, uint32_t width);
const char *mglRenderGLSLTypeName(uint32_t type);
uint32_t mglRenderGLSLMatrixCols(uint32_t type);
uint32_t mglRenderGLSLMatrixRows(uint32_t type);
const char *mglRenderGLSLColumnType(uint32_t rows);
int mglRenderGLSLNeedsFlat(uint32_t type);
int mglRenderTargetIsRenderbuffer(uint32_t target);
int mglRenderMSSamplePlaneAdjust(int in_ms_loop, uint32_t target,
                                 int32_t offset);
int mglRenderClipOriginIsLowerLeft(uint32_t origin);
int mglRenderErrorIsNone(uint32_t error);
uint32_t mglRenderErrorNone(void);
uint32_t mglRenderErrorInvalidOperation(void);
uint32_t mglRenderErrorInvalidValue(void);
uint32_t mglRenderErrorOutOfMemory(void);
uint32_t mglRenderGLBoolean(int value);
int mglRenderStopColorAttachmentScan(uint32_t next_index, uint32_t max,
                                     int next_is_none, int has_next_color);
int mglRenderEmulateTriangleFan(uint32_t mode, int polygon_point);
int mglRenderEmulateLineLoop(uint32_t mode);
int mglRenderEmulateQuads(uint32_t mode, int polygon_point);
int mglRenderFilterIsNearest(uint32_t filter);
uint32_t mglRenderNearestFilter(void);
int mglRenderCPUFormatTypeForInternalFormat(uint32_t internalformat,
                                            uint32_t *out_format,
                                            uint32_t *out_type);
int mglRenderQuadsCountTooSmall(uint32_t mode, int32_t count);
int mglRenderPolygonPointEmulateMode(uint32_t mode);
int mglRenderCompareFuncFromGL(uint32_t func, uint32_t *out);
int mglRenderFrontFaceIsClockwise(uint32_t front_face);
int mglRenderFrontFaceIsCounterClockwise(uint32_t front_face);
int mglRenderCubeMapFaceSlice(uint32_t textarget, uint32_t *out);
int mglRenderAttachmentUsesArrayLayer(uint32_t textarget);
int mglRenderPackedDepthStencilFormat(uint32_t internalformat);
int mglRenderRepairBlendSrcFactor(uint32_t *value);
int mglRenderRepairBlendDstFactor(uint32_t *value);
int mglRenderRepairBlendEquation(uint32_t *value);
uint32_t mglRenderRepairDepthFunc(uint32_t func);
uint32_t mglRenderRepairStencilFunc(uint32_t func);
int mglRenderCPUPointerUsable(const void *p);
int mglRenderShaderResourceToGLBufferType(int spvc_type);
int mglRenderUsePlainUniformBuffers(int spvc_type);
int mglRenderStructMemberInElementRange(uint32_t member_loc_off,
                                        uint32_t loc_start, uint32_t loc_end);
int mglRenderBindableLocValid(int32_t loc, uint32_t max);
int mglRenderCPUShadowReadable(const void *cpu, int64_t size);
uint64_t mglRenderClampCopyToStruct(uint64_t dest_off, uint64_t copy_size,
                                    uint64_t struct_size);
int64_t mglRenderMappedUniformSize(int spvc_type, int64_t bound, int64_t buf_size,
                                   int64_t offset, uint64_t reflected);
int32_t mglRenderPlainUniformBaseLoc(int32_t uniform_location,
                                     uint32_t location);
uint32_t mglRenderMemberOffsetInElement(uint32_t offset,
                                        uint32_t elem_byte_start);
int mglRenderMemberOffsetInStruct(uint32_t offset, uint32_t struct_size);
void mglRenderPlainUniformArrayStrides(const char *name, uint32_t type_bytes,
                                       int32_t array_stride, uint32_t *src_out,
                                       uint32_t *elem_out);
int mglRenderMetalBackingTooSmall(int64_t gl_size, uint64_t metal_length);


/* does GL primitive mode produce polygonal primitives
 * (triangles/quads) subject to glPolygonMode point/line emulation?  Matches
 * mglDrawModeProducesPolygons.  Returns 1/0. */

/* does `mode` with `indexCount` vertices produce at
 * least one drawable segment (point/line/triangle/quad)?  Matches
 * mglPrimitiveModeHasDrawableSegment.  Returns 1/0. */
/* total triangle index count for `source_vertex_count`
 * vertices arranged as quads (4/quad -> 6 indices).  Matches
 * mglQuadTriangleIndexCount; returns 0 on overflow. */
/* Align vertex stride to 4; matches mglAlignVertexStrideForMetal. */
/* double-attrib size -> MTLVertexFormat value; matches mglDoubleVertexAttribFloatFormat. */
/* Integer attrib signedness mismatch -> Int/UInt MTLVertexFormat ABI value.
 * Returns MTLVertexFormatInvalid when no CPU conversion is required. */
enum {
    MGL_ATTRIB_CONV_NONE = 0,
    MGL_ATTRIB_CONV_DOUBLE = 1,
    MGL_ATTRIB_CONV_INT_TO_FLOAT = 2,
    MGL_ATTRIB_CONV_FIXED = 3,
    MGL_ATTRIB_CONV_UINT_1010102 = 4,
    MGL_ATTRIB_CONV_UINT_10F11F11F = 5,
    MGL_ATTRIB_CONV_INTEGER_SIGN = 6,
};

uint32_t mglRenderGLTypeSizeToVertexFormat(uint32_t type, uint32_t size,
                                           int normalized);



/* FNV-1a single hash step; matches mglHashStepU64. */
/* Fixed restart-index for a type; matches the fixed branch of
 * mglPrimitiveRestartIndexForType.  1 if defined; *out set. */
/* GL uniform/attrib type -> element byte size; matches mglGLTypeElementByteSize. */

typedef struct MGLRenderGeometryGatherResult_t {
    uint32_t *gather;          /* malloc'd raw gather (vertex_ids) */
    uint32_t gather_count;
    uint32_t primitive_count;
    uint32_t max_index;
} MGLRenderGeometryGatherResult;

/* the indexed-PATCHES geometry gather — expand a raw
 * index stream (BYTE/SHORT/INT element size) into a flat vertex-id gather,
 * counting complete primitives of `last` vertices and dropping primitive
 * restarts / trailing incomplete groups.  Pure CPU; caller frees
 * result.gather.  Returns 0 on success, -1 on bad args / no valid gather. */
int mglRenderGeometryGatherIndices(
    const uint8_t *index_bytes,
    uint32_t index_type_byte_width,   /* 1, 2 or 4 */
    uint32_t count,
    int restart_enabled,
    uint32_t restart_index,
    uint32_t input_vertices,
    MGLRenderGeometryGatherResult *out);

typedef struct MGLRenderReadTextureRegionClip_t {
    int32_t copy_w;
    int32_t copy_h;
    int32_t dst_x;
    int32_t dst_y;
    int32_t metal_src_x;
    int32_t metal_src_y;
    int empty;   /* copyW <= 0 || copyH <= 0 (nothing to copy) */
} MGLRenderReadTextureRegionClip;


typedef struct MGLRenderThreadgroupSize_t {
    uint32_t x;   /* local workgroup size with 0 resolved to 1 */
    uint32_t y;
    uint32_t z;
} MGLRenderThreadgroupSize;

/* compute dispatch threadgroup size — resolves a
 * zero local workgroup component to 1 (the `x ? x : 1` default used by the
 * ObjC dispatch fallback).  Pure computation shared by both gates. */
int mglRenderThreadgroupSize(
    uint32_t local_x, uint32_t local_y, uint32_t local_z,
    MGLRenderThreadgroupSize *out);

typedef struct MGLRenderVertexAttribResolve_t {
    int use_binding_table;   /* bindingIndex < limit && binding has buffer */
    int64_t binding_offset;  /* table offset, or attrib binding_offset */
    uint32_t stride;         /* table stride, or attrib stride */
    uint32_t divisor;
} MGLRenderVertexAttribResolve;


typedef struct MGLRenderPolygonOffsetDecision_t {
    int triangle_fill_mode;      /* 0 = fill, 1 = lines */
    int needs_polygon_mode_repair;
    int enable_depth_bias;
} MGLRenderPolygonOffsetDecision;

/* polygon-offset draw decision — the triangle fill
 * mode (GL_LINE -> lines), the invalid polygon-mode repair condition and
 * the depth-bias enablement per polygon mode with the three capability
 * flags.  Pure decision shared by both gates. */
int mglRenderPolygonOffsetDecision(
    uint32_t mode,
    int has_ctx,
    int produces_polygons,
    uint32_t polygon_mode,
    int cap_point,
    int cap_line,
    int cap_fill,
    MGLRenderPolygonOffsetDecision *out);


typedef struct MGLRenderScaledBlitUVs_t {
    float uv_left;
    float uv_top;
    float uv_right;
    float uv_bottom;
} MGLRenderScaledBlitUVs;

typedef struct MGLRenderBlitScissorRect_t {
    int64_t x0;
    int64_t x1;
    int64_t y0;
    int64_t y1;
} MGLRenderBlitScissorRect;



typedef struct MGLRenderBlitFramebufferPlan_t {
    int src_x_forward;
    int src_y_forward;
    int dst_x_forward;
    int dst_y_forward;
    int blit_needs_flip;
    int needs_scaled_blit;
    double src_min_x;
    double src_max_x;
    double src_min_y;
    double src_max_y;
    double dst_min_x;
    double dst_max_x;
    double dst_min_y;
    double dst_max_y;
    double src_w;
    double src_h;
    double dst_w;
    double dst_h;
    int64_t copy_src_x;
    int64_t copy_src_y;
    int64_t copy_dst_x;
    int64_t copy_dst_y;
    int64_t copy_w;
    int64_t copy_h;
    int64_t src_metal_y;
    int64_t dst_metal_y;
    double scaled_dst_metal_y;
} MGLRenderBlitFramebufferPlan;


/* C1: MGLRenderGetTexImagePlan + mglRenderGetTexImagePlan -> mgl_readback_policy.h */

typedef struct MGLRenderLevelUploadOp_t {
    uint32_t level;
    uint32_t kind;          /* 0 = upload op, 1 = short-backing (skip) */
    uint32_t width;
    uint32_t height;
    uint64_t bytes_per_row;
    uint64_t bytes_per_image;
    uint64_t copy_depth;
    uint64_t available_bytes; /* short-backing: bytes available */
    uint64_t needed_bytes;    /* short-backing: bytes_per_image * copy_depth */
    const void *data;         /* upload op: borrowed or owned (owns_data) */
    int owns_data;
} MGLRenderLevelUploadOp;

/* build the level-upload op list for a single-face
 * (2D) CPU upload — inlines the has-uploadable CPU-data check, runs
 * mglRenderTexturePrepareLevelUpload per level and classifies each as
 * upload op / short-backing / bad.  levels must have level_count entries.
 * Returns 0 with *op_count_out ops (capacity must hold level_count), or -1
 * on bad args / capacity overflow.  short-backing ops carry kind=1 with the
 * have/need bytes; bad levels are counted in *bad_out (skipped silently,
 * matching the ObjC baseline). */
int mglRenderBuildLevelUploadOps(
    const TextureLevel *levels,
    uint32_t level_count,
    uint32_t texture_type,
    uint32_t internal_format,
    uint32_t pixel_format,
    MGLRenderLevelUploadOp *ops,
    uint32_t ops_capacity,
    uint32_t *op_count_out,
    uint32_t *short_backing_out,
    uint32_t *bad_out);

typedef struct MGLRenderLevelUploadPrep_t {
    const void *data;         /* borrowed or owned */
    uint64_t bytes_per_row;
    uint64_t bytes_per_image;
    uint64_t copy_depth;
    uint64_t available_bytes;
    int owns_data;            /* 1: caller must free((void *)data) */
} MGLRenderLevelUploadPrep;



/* R8 swizzle component + single-channel upload expand.
 * Resolve mirrors mglResolveR8SwizzledComponent (tex unused).  Create
 * expands GL_R8 1B/px → RGBA8 via the four swizzle enums; malloc'd
 * result, NULL on bad args / non-R8 / size cap. */
uint8_t mglRenderResolveR8SwizzledComponent(uint32_t swizzle, uint8_t red);
/* stored color-component count for an internal format.
 * Mirrors mglStoredColorComponentsForTexture after the null-tex check
 * (null stays in ObjC and returns 4).  Unknown formats → 4. */
uint32_t mglRenderStoredColorComponents(uint32_t internal_format);

enum {
    MGL_RENDER_PIPELINE_CACHE_KEY_WORDS = 7,
    MGL_RENDER_PIPELINE_COLOR_ATTACHMENTS = 8,
};
#define MGL_RENDER_PIPELINE_CACHE_KEY_WORDS_DEFINED 1

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


/* 1 if the default Metal device is Apple Paravirtual (hosted CI VMs). */


typedef struct MGLRenderPipelineActiveState_t {
    void *pipeline_state;
    void *vertex_function;
    void *fragment_function;
    uint32_t color0_format;
    uint32_t depth_format;
    uint32_t stencil_format;
    uint32_t program_name;
} MGLRenderPipelineActiveState;

typedef struct MGLRenderPipelineBlendState_t {
    uint32_t source_rgb_factor;
    uint32_t destination_rgb_factor;
    uint32_t source_alpha_factor;
    uint32_t destination_alpha_factor;
    uint32_t rgb_operation;
    uint32_t alpha_operation;
    uint32_t color_write_mask;
} MGLRenderPipelineBlendState;

int mglRenderSerializeBinaryArchive(void *binary_archive,
                                       void *url,
                                       char *err,
                                       size_t errcap);
int mglRenderSetVisibilityResultMode(void *render_encoder,
                                        uint32_t mode,
                                        uint64_t offset);
int mglRenderSetVisibilityResultModeForRenderEncoderOwner(MGLRenderEncoderOwner *render_encoder_owner, uint32_t mode, uint64_t offset);
int mglRenderSampleTimestamps(uint64_t *cpu_timestamp_out,
                                 uint64_t *gpu_timestamp_out);


enum {
    MGL_RENDER_AUX_COMPUTE_SCALED_BLIT = 1,
    MGL_RENDER_AUX_COMPUTE_MSAA_INTEGER_RESOLVE = 2,
    MGL_RENDER_AUX_RENDER_SCALED_BLIT = 3,
    MGL_RENDER_AUX_RENDER_SCALED_DEPTH_BLIT = 4,
    MGL_RENDER_AUX_RENDER_CLEAR_RECT = 5,
    MGL_RENDER_AUX_COMPUTE_GS_XFB_SCATTER = 6,
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



enum {
    MGL_RENDER_BINDING_STAGE_VERTEX = 0,
    MGL_RENDER_BINDING_STAGE_FRAGMENT = 1,
};

enum {
    MGL_RENDER_BINDING_VERTEX_TEXTURE = 0,
    MGL_RENDER_BINDING_FRAGMENT_TEXTURE = 1,
    MGL_RENDER_BINDING_VERTEX_SAMPLER = 2,
    MGL_RENDER_BINDING_FRAGMENT_SAMPLER = 3,
    MGL_RENDER_BINDING_VIEWPORT = 4,
    MGL_RENDER_BINDING_SCISSOR = 5,
    MGL_RENDER_BINDING_TRIANGLE_FILL = 6,
    MGL_RENDER_BINDING_SETTER_COUNT = 7,
};

typedef struct MGLRenderBindingStats {
    uint64_t emitted[MGL_RENDER_BINDING_SETTER_COUNT];
    uint64_t skipped[MGL_RENDER_BINDING_SETTER_COUNT];
} MGLRenderBindingStats;


int mglRenderSetComputeBytes(void *compute_encoder,
                                const void *bytes,
                                size_t length,
                                uint32_t index);
int mglRenderSetComputeThreadgroupMemoryLength(void *compute_encoder,
                                                  uint64_t length,
                                                  uint32_t index);

/* Compute binding snapshot, structurally equivalent to the render snapshot.
 * Kinds select buffer, inline bytes, texture, or sampler operations. The
 * caller validates inputs; malformed operations return -1. Temporary bridged
 * objects must be flushed immediately and must not enter deferred replay. */
#define MGL_RENDER_COMPUTE_BINDING_SNAPSHOT_MAX_OPS 32u

typedef struct MGLRenderComputeBindingOp_t {
    uint32_t kind;      /* 0 = buffer, 1 = bytes, 2 = texture, 3 = sampler */
    uint32_t index;     /* Metal slot */
    uint64_t offset;    /* kind 0: byte offset */
    void *buffer;       /* kind 0/2/3: borrowed MTL object (NULL = clear) */
    const void *bytes;  /* kind 1: borrowed byte pointer */
    uint32_t length;    /* kind 1: byte length */
} MGLRenderComputeBindingOp;

typedef struct MGLRenderComputeBindingSnapshot_t {
    uint32_t op_count;
    MGLRenderComputeBindingOp
        ops[MGL_RENDER_COMPUTE_BINDING_SNAPSHOT_MAX_OPS];
} MGLRenderComputeBindingSnapshot;


/* Value-state compute dispatch plan. A zero local dimension resolves to one.
 * The C++ backend encodes direct or indirect dispatch from this plan. */
#define MGL_RENDER_COMPUTE_DISPATCH_DIRECT   0
#define MGL_RENDER_COMPUTE_DISPATCH_INDIRECT 1

typedef struct MGLRenderComputePlan_t {
    uint32_t dispatch_kind;   /* DIRECT / INDIRECT */
    uint32_t groups_x;
    uint32_t groups_y;
    uint32_t groups_z;
    uint32_t local_x;         /* Zero resolves to one. */
    uint32_t local_y;
    uint32_t local_z;
    void *indirect_buffer;    /* INDIRECT: borrowed MTL::Buffer* */
    uint64_t indirect_offset; /* Byte offset of the indirect argument block. */
} MGLRenderComputePlan;


/*  compute execution plan: ObjC collects the ordered binding operations
 * and keeps temporary Metal objects alive until this call returns. C++ owns
 * encoder creation, pipeline/binding replay, dispatch, and endEncoding. */
/* Per-patch TES compute expansion emits one dispatch (+ contract bytes
 * binding) per input patch. Cull-distance isoline grids reach ~144 patches;
 * barrier CTS uses 1024 patches (2048 result verts). Keep headroom above
 * the old 128/512 caps that returned GL_INVALID_OPERATION once patchCount
 * exceeded them. */
#define MGL_RENDER_COMPUTE_EXECUTION_MAX_OPS 4096u
#define MGL_RENDER_COMPUTE_EXECUTION_MAX_DISPATCHES 2048u

typedef struct MGLRenderComputeDispatchEntry_t {
    /* Replay this dispatch after exactly binding_op_count binding operations. */
    uint32_t binding_op_count;
    MGLRenderComputePlan dispatch;
} MGLRenderComputeDispatchEntry;

typedef struct MGLRenderComputeExecutionPlan_t {
    void *pipeline; /* +0 borrowed MTL::ComputePipelineState* */
    uint32_t binding_op_count;
    MGLRenderComputeBindingOp
        binding_ops[MGL_RENDER_COMPUTE_EXECUTION_MAX_OPS];
    uint32_t dispatch_op_count;
    MGLRenderComputeDispatchEntry
        dispatch_ops[MGL_RENDER_COMPUTE_EXECUTION_MAX_DISPATCHES];
    /* Backward-compatible single-dispatch form used when dispatch_op_count=0. */
    MGLRenderComputePlan dispatch;
    /* Barrier emitted after the final dispatch, before endEncoding. */
    uint32_t barrier_scope;
    /* Barrier emitted *between* consecutive dispatches of this encoder.
     *
     * Metal only guarantees that dispatches inside one encoder execute in
     * submission order; memory written by dispatch N is not visible to
     * dispatch N+1 unless a memoryBarrier separates them.  Leaving this NONE
     * therefore asserts that the dispatches are memory-independent: each
     * writes a range disjoint from the others and reads only data produced on
     * the host or by an earlier encoder (an encoder boundary orders memory).
     * The per-patch TES expansion (mglTessAppendEvalPerPatchDispatches) is
     * such a plan and documents the invariant there.
     *
     * A plan whose later dispatches consume earlier dispatches' writes must
     * set this to the touched resource classes;
     * mglRenderEncodeComputeExecutionPlanForCommandBufferOwner then fences
     * every dispatch boundary instead of silently racing. */
    uint32_t dispatch_barrier_scope;
} MGLRenderComputeExecutionPlan;

/* Value-state barrier request. These values intentionally mirror Metal's
 * BarrierScope bit values without exposing MTL::* through the C ABI. */
enum {
    MGL_RENDER_COMPUTE_BARRIER_NONE = 0u,
    MGL_RENDER_COMPUTE_BARRIER_BUFFERS = 1u,
    MGL_RENDER_COMPUTE_BARRIER_TEXTURES = 2u,
    MGL_RENDER_COMPUTE_BARRIER_RENDER_TARGETS = 4u,
};



/* Fixed GS/TES compute-dispatch setup. The backend creates the encoder and
 * binds pipeline ABI slots; GL stage resources are bound through the C++
 * facade between begin and end. */
#define MGL_RENDER_COMPUTE_DISPATCH_MAX_BUFFERS 16u
#define MGL_RENDER_COMPUTE_DISPATCH_MAX_BYTES 4u

typedef struct MGLRenderComputeBufferEntry_t {
    void *buffer;   /* +0 borrowed MTL::Buffer* */
    uint64_t offset;
    uint32_t index;
} MGLRenderComputeBufferEntry;

typedef struct MGLRenderComputeBytesEntry_t {
    const void *bytes;
    uint32_t length;
    uint32_t index;
} MGLRenderComputeBytesEntry;

typedef struct MGLRenderComputeDispatchSetup_t {
    void *pipeline;             /* +0 borrowed MTL::ComputePipelineState* */
    uint32_t buffer_count;
    MGLRenderComputeBufferEntry
        buffers[MGL_RENDER_COMPUTE_DISPATCH_MAX_BUFFERS];
    uint32_t bytes_count;
    MGLRenderComputeBytesEntry
        bytes[MGL_RENDER_COMPUTE_DISPATCH_MAX_BYTES];
} MGLRenderComputeDispatchSetup;



enum {
    MGL_RENDER_ERROR_DOMAIN_CAPACITY = 128,
    MGL_RENDER_ERROR_DESCRIPTION_CAPACITY = 512,
};

typedef struct MGLRenderCommandBufferState_t {
    uint32_t status;
    uint32_t has_error;
    int64_t error_code;
    char error_domain[MGL_RENDER_ERROR_DOMAIN_CAPACITY];
    char error_description[MGL_RENDER_ERROR_DESCRIPTION_CAPACITY];
} MGLRenderCommandBufferState;

typedef enum MGLRenderCommandBufferCommitAction_t {
    MGL_RENDER_COMMAND_BUFFER_COMMIT_PROCEED = 0,
    MGL_RENDER_COMMAND_BUFFER_COMMIT_SKIP_ALREADY_COMMITTED = 1,
} MGLRenderCommandBufferCommitAction;

typedef struct MGLRenderCommandBufferCommitDecision_t {
    uint32_t action;
} MGLRenderCommandBufferCommitDecision;

typedef enum MGLRenderCommandBufferTransactionResult_t {
    MGL_RENDER_COMMAND_BUFFER_TRANSACTION_COMMITTED = 0,
    MGL_RENDER_COMMAND_BUFFER_TRANSACTION_SKIPPED = 1,
    MGL_RENDER_COMMAND_BUFFER_TRANSACTION_NESTED = 2,
    MGL_RENDER_COMMAND_BUFFER_TRANSACTION_ERROR = 3,
} MGLRenderCommandBufferTransactionResult;

typedef struct MGLRenderCommandRecoverySnapshot_t {
    uint64_t consecutive_errors;
    uint64_t consecutive_successes;
    double last_error_time;
    uint32_t recovery_mode;
} MGLRenderCommandRecoverySnapshot;

/* Result of one owner-aware submit transaction.  State snapshots are value
 * copies; no command-buffer pointer is retained by the result. */
typedef struct MGLRenderCommandBufferTransaction_t {
    MGLRenderCommandBufferState before;
    MGLRenderCommandBufferState after;
    MGLRenderCommandBufferState completion;
    uint32_t result;
    uint32_t used_submission;
    uint32_t completion_registered;
    uint32_t waited;
    uint32_t has_error;
    uint32_t is_driver_rejection;
    uint32_t device_reset_requested;
    uint32_t recovery_error_recorded;
    MGLRenderCommandRecoverySnapshot recovery;
    uint32_t needs_new_command_buffer;
    /* Set when the C++ owner created the next current command buffer as part
     * of this transaction.  A zero value means the caller must retain its
     * legacy queue/reset adapter (for example an adopted ObjC buffer). */
    uint32_t current_command_buffer_created;
} MGLRenderCommandBufferTransaction;

/* Result of one owner-contained compute execution.  No submission or Metal
 * object pointer escapes the transaction.  When submitted is zero, the
 * encoded compute work remains in CommandBufferOwner.current for the normal
 * renderer flush. */
typedef struct MGLRenderComputeExecutionResult_t {
    MGLRenderCommandBufferTransaction transaction;
    uint32_t submitted;
    uint32_t cpu_prefix_synchronized;
    uint32_t failed_copy_back_index;
} MGLRenderComputeExecutionResult;


typedef struct MGLRenderCommandBufferCompletionDecision_t {
    uint32_t has_error;
    uint32_t is_driver_rejection;
} MGLRenderCommandBufferCompletionDecision;

typedef struct MGLRenderCommandRecoverySuccess_t {
    MGLRenderCommandRecoverySnapshot state;
    uint32_t sustained_recovery;
    uint64_t recovered_successes;
    uint64_t previous_errors;
} MGLRenderCommandRecoverySuccess;

typedef struct MGLRenderCommandRecoverySkipDecision_t {
    MGLRenderCommandRecoverySnapshot state;
    uint32_t should_skip;
    uint32_t entered_recovery_mode;
    uint32_t recovery_timed_out;
    uint64_t previous_errors;
} MGLRenderCommandRecoverySkipDecision;

typedef struct MGLRenderCommandBufferCompletionResult_t {
    MGLRenderCommandBufferCompletionDecision decision;
    MGLRenderCommandRecoverySnapshot state;
    uint32_t sustained_recovery;
    uint32_t cleared_recovery_mode;
    uint64_t recovered_successes;
    uint64_t previous_errors;
} MGLRenderCommandBufferCompletionResult;

typedef void (*MGLRenderCommandBufferCompletion)(
    void *context,
    const MGLRenderCommandBufferState *state);
typedef void (*MGLRenderDestroyContext)(void *context);


#include "mgl_render_pass_plan.h"

/* Exact, program-based replacement for the retired shader-source scans: reads
 * the per-stage builtin mask published by the frontend. */
int mglRenderFragmentNeedsPerSampleMSValues(const Program *program);


void mglRenderFillFragCoordSlot(int use_fragcoord, int use_sample,
                                uint32_t pass_height, int lower_left,
                                uint32_t num_samples, uint32_t sample_buffers,
                                int ms_loop, uint32_t forced_sample_id,
                                float out[4]);

void mglRenderClampLodBiasArray(float *bias, uint32_t count, float biasmax);

typedef struct MGLDirtyDomainPlan {
    int has_dirty;
    int sync_render_pass;
    int bind_fbo_attachments;
    int remap_buffers;
    int defer_buffer_map;
    int bind_textures;
    int vao_path;
    int buffer_path;
    int render_state_path;
    int sync_pipeline;
    int dirty_buffer_data;
} MGLDirtyDomainPlan;



uint64_t mglXfbAdvanceWriteOffset(uint64_t current, uint64_t written);
int mglRenderAllocateMDIScratch(MGLMDIScratchOwner *owner, uint64_t length, uint64_t alignment, void **buffer_out, uint64_t *offset_out, uint64_t *capacity_out);

/* Per-draw binding snapshot. GL-side deduplication and accounting determine
 * which bindings are emitted; the backend replays the resulting operation list. */
#define MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS 32u

/* One per-draw binding op: kind 0 = set buffer (buffer == NULL clears the
 * slot, matching mglRenderSetRenderBuffer with a nil resource), kind 1 =
 * set bytes (bytes borrowed — valid until EncodeBindingSnapshot returns).
 * The op list keeps the exact per-stage emit order, including interleaved
 * buffer/bytes/clear ops on the same slot. */
typedef struct MGLRenderBindingOp_t {
    uint32_t kind;      /* 0 = buffer, 1 = bytes */
    uint32_t index;     /* Metal slot */
    uint64_t offset;    /* kind 0: byte offset */
    void *buffer;       /* kind 0: borrowed MTL::Buffer* (NULL = clear) */
    const void *bytes;  /* kind 1: borrowed byte pointer */
    uint32_t length;    /* kind 1: byte length */
} MGLRenderBindingOp;

typedef struct MGLRenderBindingSnapshot_t {
    uint32_t vertex_op_count;
    MGLRenderBindingOp
        vertex_ops[MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS];
    uint32_t fragment_op_count;
    MGLRenderBindingOp
        fragment_ops[MGL_RENDER_BINDING_SNAPSHOT_MAX_OPS];
} MGLRenderBindingSnapshot;


/* Texture/sampler bindings are collected separately from buffer/bytes ops.
 * Resource resolution may rotate the render encoder while uploading a
 * texture, so ObjC submits this snapshot in ordered segments at those
 * boundaries.  The C++ binding owner remains authoritative for dedup state
 * and retains every resource that becomes current. */
#define MGL_RENDER_RESOURCE_BINDING_SNAPSHOT_MAX_OPS 512u

enum {
    MGL_RENDER_RESOURCE_BINDING_TEXTURE = 0,
    MGL_RENDER_RESOURCE_BINDING_SAMPLER = 1,
};

typedef struct MGLRenderResourceBindingOp_t {
    uint32_t kind;
    uint32_t index;
    void *resource; /* borrowed MTL::Texture* or MTL::SamplerState* */
} MGLRenderResourceBindingOp;

typedef struct MGLRenderResourceBindingSnapshot_t {
    uint32_t vertex_op_count;
    MGLRenderResourceBindingOp
        vertex_ops[MGL_RENDER_RESOURCE_BINDING_SNAPSHOT_MAX_OPS];
    uint32_t fragment_op_count;
    MGLRenderResourceBindingOp
        fragment_ops[MGL_RENDER_RESOURCE_BINDING_SNAPSHOT_MAX_OPS];
} MGLRenderResourceBindingSnapshot;


/* Replays a simple draw batch in C++. Eligible batches contain no dynamic
 * bindings, sampler snapshots, cull-distance state, polygon emulation, or
 * primitive restart. The command array remains a read-only arena snapshot. */
#define MGL_RENDER_REPLAY_BATCH_MAX_COMMANDS 128u

typedef struct MGLRenderReplayBatchCommand_t {
    uint32_t cmd_type;          /* MGLDrawCommandType value. */
    int32_t first;
    uint32_t count;
    uint32_t instance_count;
    int32_t base_vertex;
    uint32_t base_instance;
    uint32_t index_type;        /* Converted MTLIndexType value. */
    uint32_t index_buffer_offset;
    void *index_buffer;         /* Borrowed prepared MTL::Buffer*. */
} MGLRenderReplayBatchCommand;

typedef struct MGLRenderReplayBatch_t {
    uint32_t primitive_type;    /* MTLPrimitiveType（batch key） */
    uint32_t command_count;
    const MGLRenderReplayBatchCommand *commands;
} MGLRenderReplayBatch;

enum {
    MGL_RENDER_REPLAY_BATCH_OK = 0,
    MGL_RENDER_REPLAY_BATCH_NEEDS_OBJC = 1,
    MGL_RENDER_REPLAY_BATCH_ERROR = -1,
};

/* The caller validates command kinds, index buffers, index types, and limits.
 * A non-success result requires replaying the entire batch through the caller;
 * partial fallback is not allowed. */
int mglRenderReplayBatchDraws(void *render_encoder,
                                 const MGLRenderReplayBatch *batch,
                                 char *err,
                                 size_t errcap);

enum {
    MGL_RENDER_MAX_COLOR_ATTACHMENTS = 8,
    MGL_RENDER_MAX_SAMPLE_POSITIONS = 32,
};

typedef struct MGLRenderPassAttachmentState_t {
    void *texture;
    void *resolve_texture;
    uint64_t level;
    uint64_t slice;
    uint64_t depth_plane;
    uint64_t resolve_level;
    uint64_t resolve_slice;
    uint64_t resolve_depth_plane;
    uint32_t load_action;
    uint32_t store_action;
    uint32_t layered;
    uint32_t _padding;
    uint64_t store_action_options;
} MGLRenderPassAttachmentState;

typedef struct MGLRenderPassColorState_t {
    MGLRenderPassAttachmentState attachment;
    double clear_red;
    double clear_green;
    double clear_blue;
    double clear_alpha;
} MGLRenderPassColorState;

typedef struct MGLRenderPassDepthState_t {
    MGLRenderPassAttachmentState attachment;
    double clear_depth;
    uint32_t resolve_filter;
} MGLRenderPassDepthState;

typedef struct MGLRenderPassStencilState_t {
    MGLRenderPassAttachmentState attachment;
    uint32_t clear_stencil;
    uint32_t resolve_filter;
} MGLRenderPassStencilState;

typedef struct MGLRenderSamplePosition_t {
    float x;
    float y;
} MGLRenderSamplePosition;

typedef struct MGLRenderPassState_t {
    MGLRenderPassColorState
        color[MGL_RENDER_MAX_COLOR_ATTACHMENTS];
    MGLRenderPassDepthState depth;
    MGLRenderPassStencilState stencil;
    void *visibility_result_buffer;
    void *rasterization_rate_map;
    uint64_t render_target_array_length;
    uint64_t render_target_width;
    uint64_t render_target_height;
    uint64_t default_raster_sample_count;
    uint64_t imageblock_sample_length;
    uint64_t threadgroup_memory_length;
    uint64_t tile_width;
    uint64_t tile_height;
    uint32_t visibility_result_type;
    uint32_t support_color_attachment_mapping;
    uint32_t sample_position_count;
    MGLRenderSamplePosition
        sample_positions[MGL_RENDER_MAX_SAMPLE_POSITIONS];
} MGLRenderPassState;


typedef enum MGLRenderPassAttachmentKind_t {
    MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR = 0,
    MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH = 1,
    MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL = 2,
} MGLRenderPassAttachmentKind;

int mglRenderPassAttachmentClass(uint32_t kind);
int mglRenderPassColorAttachmentIndexValid(uint32_t color_index,
                                           uint32_t max_color);

typedef struct MGLRenderPassIdentityState_t {
    void *framebuffer;
    uint32_t framebuffer_name;
    uint32_t draw_buffer;
    uint32_t draw_buffer_count;
    uint32_t draw_buffers[MGL_RENDER_MAX_COLOR_ATTACHMENTS];
} MGLRenderPassIdentityState;

typedef struct MGLRenderFboMatchCacheState_t {
    uint32_t fbo_name;
    uint64_t generation;
    int result;
} MGLRenderFboMatchCacheState;

int mglRenderSetFboMatchCache(MGLRenderPassIdentityOwner *owner, const MGLRenderFboMatchCacheState *cache);

int mglRenderPendingEventPrepare(MGLPendingEventOwner *owner_handle, int sync_name, void **event_out);
int mglRenderPendingEventDetach(MGLPendingEventOwner *owner_handle, int *sync_name_out, void **event_out);

typedef struct MGLRenderBufferCopyEntry_t {
    void *source_buffer;
    uint64_t source_offset;
    void *destination_buffer;
    uint64_t destination_offset;
    uint64_t length;
} MGLRenderBufferCopyEntry;
/* Encode and end a complete buffer-to-texture upload blit in C++. The
 * command buffer retains the encoded resources after this function returns. */
int mglRenderEncodeTextureUpload(void *command_buffer,
                                    void *source_buffer,
                                    uint64_t source_offset,
                                    uint64_t source_bytes_per_row,
                                    uint64_t source_bytes_per_image,
                                    uint64_t source_width,
                                    uint64_t source_height,
                                    uint64_t source_depth,
                                    void *destination_texture,
                                    uint64_t destination_slice,
                                    uint64_t destination_level,
                                    uint64_t destination_x,
                                    uint64_t destination_y,
                                    uint64_t destination_z);
/* Multi-slice form used by array-texture subimages.  Arithmetic and resource
 * extents are validated before a single blit encoder is opened, so a bad
 * range cannot leave a partially encoded layer prefix. */
int mglRenderEncodeTextureUploadLayers(
    void *command_buffer,
    void *source_buffer,
    uint64_t source_offset,
    uint64_t source_bytes_per_row,
    uint64_t source_bytes_per_image,
    uint64_t source_layer_stride,
    uint64_t source_width,
    uint64_t source_height,
    uint64_t source_depth,
    void *destination_texture,
    uint64_t destination_base_slice,
    uint64_t layer_count,
    uint64_t destination_level,
    uint64_t destination_x,
    uint64_t destination_y,
    uint64_t destination_z);
int mglRenderEncodeTextureUploadLayersForCommandBufferOwner(MGLCommandBufferOwner *command_buffer_owner, void *source_buffer, uint64_t source_offset, uint64_t source_bytes_per_row, uint64_t source_bytes_per_image, uint64_t source_layer_stride, uint64_t source_width, uint64_t source_height, uint64_t source_depth, void *destination_texture, uint64_t destination_base_slice, uint64_t layer_count, uint64_t destination_level, uint64_t destination_x, uint64_t destination_y, uint64_t destination_z);
int mglRenderBlitCopyBufferToTexture(void *blit_encoder,
                                        void *source_buffer,
                                        uint64_t source_offset,
                                        uint64_t source_bytes_per_row,
                                        uint64_t source_bytes_per_image,
                                        uint64_t source_width,
                                        uint64_t source_height,
                                        uint64_t source_depth,
                                        void *destination_texture,
                                        uint64_t destination_slice,
                                        uint64_t destination_level,
                                        uint64_t destination_x,
                                        uint64_t destination_y,
                                        uint64_t destination_z);
int mglRenderBlitCopyTexture(void *blit_encoder,
                                void *source_texture,
                                uint64_t source_slice,
                                uint64_t source_level,
                                uint64_t source_x,
                                uint64_t source_y,
                                uint64_t source_z,
                                uint64_t width,
                                uint64_t height,
                                uint64_t depth,
                                void *destination_texture,
                                uint64_t destination_slice,
                                uint64_t destination_level,
                                uint64_t destination_x,
                                uint64_t destination_y,
                                uint64_t destination_z);
int mglRenderBlitCopyTextureToBuffer(
    void *blit_encoder,
    void *source_texture,
    uint64_t source_slice,
    uint64_t source_level,
    uint64_t source_x,
    uint64_t source_y,
    uint64_t source_z,
    uint64_t width,
    uint64_t height,
    uint64_t depth,
    void *destination_buffer,
    uint64_t destination_offset,
    uint64_t destination_bytes_per_row,
    uint64_t destination_bytes_per_image);


/* Unified value-state draw plan. Resources are borrowed and final draw
 * encoding is owned by the C++ backend. */
typedef struct MGLRenderDrawPlan_t {
    uint32_t kind;              /* MGL_RENDER_DRAW_* */
    uint32_t primitive_type;    /* MTLPrimitiveType ABI value. */
    /* ARRAY: */
    uint64_t vertex_start;
    uint64_t vertex_count;
    /* INDEXED: */
    uint64_t index_count;
    uint32_t index_type;        /* MTLIndexType ABI value. */
    void *index_buffer;         /* +0 borrowed MTL::Buffer* */
    uint64_t index_buffer_offset;
    int64_t base_vertex;
    /* INDIRECT: */
    void *indirect_buffer;      /* +0 borrowed MTL::Buffer* */
    uint64_t indirect_buffer_offset;
    /* PATCHES（native TES）: */
    uint64_t control_point_count;
    uint64_t patch_start;
    uint64_t patch_count;
    void *patch_index_buffer;           /* +0 borrowed */
    uint64_t patch_index_buffer_offset;
    void *control_point_index_buffer;   /* +0 borrowed */
    uint64_t control_point_index_buffer_offset;
    /* Common fields. */
    uint64_t instance_count;
    uint64_t base_instance;
} MGLRenderDrawPlan;

enum {
    MGL_RENDER_DRAW_ARRAY = 1,
    MGL_RENDER_DRAW_INDEXED = 2,
    MGL_RENDER_DRAW_ARRAY_INDIRECT = 3,
    MGL_RENDER_DRAW_INDEXED_INDIRECT = 4,
    MGL_RENDER_DRAW_PATCHES = 5,
    MGL_RENDER_DRAW_INDEXED_PATCHES = 6,
};


typedef struct MGLRenderCullDistancePrimitive_t {
    uint32_t vertices[4];
    uint32_t vertex_count;
    uint32_t primitive_type;
    uint32_t index_count;
    uint64_t index_buffer_offset;
} MGLRenderCullDistancePrimitive;

/* Build a UInt32 index buffer whose records each represent one complete GL
 * primitive. The opaque owner retains the borrowed index buffer and the
 * per-primitive explicit vertex IDs used by exact gl_CullDistance emulation. */
int mglRenderCreateCullDistanceIndexPlan(void *device, const void *source_indices, uint32_t source_index_type, uint64_t source_index_count, uint32_t draw_mode, int primitive_restart_enabled, uint32_t primitive_restart_index, int64_t base_vertex, int polygon_line_mode, MGLCullDistanceIndexPlan **owner_out, void **index_buffer_out, uint64_t *primitive_count_out);



typedef struct MGLCullDistanceEmuParams_t {
    uint32_t prim_vertex_count;
    uint32_t culldist_offset;
    uint32_t vertex_stride;
    uint32_t culldist_size;
    uint32_t first_vertex;
    uint32_t explicit_vertex_count;
    uint32_t explicit_vertices[4];
    uint32_t first_instance;
    uint32_t instance_stride;
} MGLCullDistanceEmuParams;

typedef struct MGLRenderCullDistanceLayout_t {
    void *mtl_buffer;
    int64_t binding_offset;
    uint32_t stride;
    int64_t first_relative_offset;
    uint32_t culldist_size;
} MGLRenderCullDistanceLayout;


/* O1.3: ObjC fills VAO-resolved ports; C++ builds layout (+ optional dummy). */
typedef struct MGLRenderCullDistanceAttribPort {
    void *mtl_buffer; /* NULL / invalid → skipped */
    int64_t binding_offset;
    uint32_t stride;
    int64_t relativeoffset;
    uint8_t valid;
} MGLRenderCullDistanceAttribPort;



int mglRenderSetRenderBytes(void *render_encoder,
                               const void *bytes,
                               size_t length,
                               uint32_t stage,
                               uint32_t index);
int mglRenderSetRenderDepthStencilState(void *render_encoder,
                                           void *depth_stencil_state);
int mglRenderSetRenderViewport(void *render_encoder,
                                  double origin_x,
                                  double origin_y,
                                  double width,
                                  double height,
                                  double znear,
                                  double zfar);
int mglRenderSetRenderScissor(void *render_encoder,
                                 uint64_t x,
                                 uint64_t y,
                                 uint64_t width,
                                 uint64_t height);
int mglRenderSetDepthClipMode(void *render_encoder, uint32_t mode);
int mglRenderSetStencilReferenceValues(void *render_encoder,
                                          uint32_t front_reference,
                                          uint32_t back_reference);
int mglRenderSetRenderBytesForOwner(MGLRenderEncoderOwner *render_encoder_owner, const void *bytes, size_t length, uint32_t stage, uint32_t index);
int mglRenderSetRenderDepthStencilStateForOwner(MGLRenderEncoderOwner *render_encoder_owner, void *depth_stencil_state);
int mglRenderSetRenderViewportForOwner(MGLRenderEncoderOwner *render_encoder_owner, double origin_x, double origin_y, double width, double height, double znear, double zfar);
int mglRenderSetRenderScissorForOwner(MGLRenderEncoderOwner *render_encoder_owner, uint64_t x, uint64_t y, uint64_t width, uint64_t height);
int mglRenderSetDepthClipModeForOwner(MGLRenderEncoderOwner *render_encoder_owner, uint32_t mode);
int mglRenderSetStencilReferenceValuesForOwner(MGLRenderEncoderOwner *render_encoder_owner, uint32_t front_reference, uint32_t back_reference);

int mglRenderUseRenderResource(void *render_encoder,
                                  void *resource,
                                  uint32_t usage,
                                  uint32_t stages);
int mglRenderExecuteIndirectCommands(void *render_encoder,
                                        void *indirect_buffer,
                                        uint64_t location,
                                        uint64_t length);
int mglRenderReplayBatchDrawsForRenderEncoderOwner(MGLRenderEncoderOwner *render_encoder_owner, const MGLRenderReplayBatch *batch, char *err, size_t errcap);
int mglRenderUseRenderResourceForOwner(MGLRenderEncoderOwner *render_encoder_owner, void *resource, uint32_t usage, uint32_t stages);
int mglRenderExecuteIndirectCommandsForOwner(MGLRenderEncoderOwner *render_encoder_owner, void *indirect_buffer, uint64_t location, uint64_t length);

#include "mgl_render_api_command.h"
#include "mgl_render_api_binding.h"
#include "mgl_render_api_texture.h"
#include "mgl_render_api_buffer.h"
#include "mgl_render_api_pixel.h"
#include "mgl_render_api_readback.h"
#include "mgl_render_api_draw.h"
#include "mgl_render_api_query.h"
#include "mgl_render_api_lifecycle.h"

#ifdef __cplusplus
} // extern "C"
#endif
#endif
