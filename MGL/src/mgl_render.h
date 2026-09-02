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

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "mgl_render_values.h"

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
    uint32_t bug_3d_copy_from_buffer_slice_oob;
    uint32_t bug_async_shader_compile_in_vm;
    uint32_t bug_msl_pipeline_rejection;
    uint64_t command_buffer_recovery_limit;
    uint64_t max_concurrent_command_buffers;
    uint64_t texture_alignment_bytes;
    uint32_t conservative_cpu_cache_mode;
} MGLRenderCapabilityState;

#ifdef __cplusplus
extern "C" {
#endif

/* Pure synchronization helpers. Metal descriptor inspection is confined to
 * the Metal-cpp implementation TU; the C ABI carries only opaque handles and
 * integer enum values. */
bool mglRenderPassAttachmentMatchesSubresource(
    const void *descriptor,
    const MGLMetalAttachmentSubresource *subresource);
const char *mglRenderCommandBufferStatusName(uint32_t status);
const char *mglRenderLoadActionName(uint32_t action);
const char *mglRenderStoreActionName(uint32_t action);

/* Initializes the renderer. objc_device is an existing id<MTLDevice>; the C++
 * side retains it without transferring ownership. Returns 0 on success. */
int mglRenderInit(void* objc_device);

/* Releases renderer-owned MTL::* objects, including the retained device.
 * This operation is idempotent. */
void mglRenderShutdown(void);

/* Renderer initialization state as a C ABI value, never a borrowed object. */
int mglRenderIsInitialized(void);

/* Query device capabilities through Metal-cpp and return a pure value-state
 * snapshot. The device pointer is borrowed for the duration of the call. */
int mglRenderQueryCapability(void *device,
                                MGLRenderCapabilityState *state_out);

/* Load the AIR entry point named "main" with the renderer-owned device.
 * Returned library/function objects are +1 retained for the caller. */
int mglRenderLoadAIRMainFunction(const unsigned char *bytes,
                                    size_t size,
                                    void **library_out,
                                    void **function_out,
                                    char *err,
                                    size_t errcap);

/* Direct renderer entries. Objects passed here carry the +1 bridge reference
 * owned by the GL state. */
void mglRenderDeleteMTLObj(GLMContext glm_ctx, void *object);
void mglRenderReleaseBufferMetalData(GLMContext glm_ctx, Buffer *buffer);
void mglRenderReleaseBufferCowPool(Buffer *buffer);
void mglRenderBindBuffer(GLMContext glm_ctx, Buffer *buffer);
void mglRenderBufferSubData(GLMContext glm_ctx,
                               Buffer *buffer,
                               size_t offset,
                               size_t size,
                               const void *bytes);
void *mglRenderMapUnmapBuffer(GLMContext glm_ctx,
                                 Buffer *buffer,
                                 size_t offset,
                                 size_t size,
                                 unsigned int access,
                                 bool map);
void mglRenderReadBackBuffer(GLMContext glm_ctx,
                                Buffer *buffer,
                                size_t offset,
                                size_t size);
void mglRenderFlushBufferRange(GLMContext glm_ctx,
                                  Buffer *buffer,
                                  intptr_t offset,
                                  intptr_t length);
void mglRenderBindProgram(GLMContext glm_ctx, Program *program);
void mglRenderGetSync(GLMContext glm_ctx, Sync *sync);
void mglRenderWaitForSync(GLMContext glm_ctx, Sync *sync);
unsigned int mglRenderGetSyncStatus(GLMContext glm_ctx, Sync *sync);
void mglRenderReleaseSync(GLMContext glm_ctx, Sync *sync);
void mglRenderFlush(GLMContext glm_ctx, bool finish);
void mglRenderInvalidateRenderPass(GLMContext glm_ctx);
uint64_t mglRenderGetGPUTimestamp(GLMContext glm_ctx);

/* Publish the borrowed runtime owner handles used by direct C++ callbacks. */
int mglRenderAttachRuntimeOwners(GLMContext glm_ctx,
                                    void *command_buffer_owner,
                                    void *render_encoder_owner,
                                    void *render_pass_state_owner);
void mglRenderDetachRuntimeOwners(GLMContext glm_ctx);
void mglRenderBeginTimerQueryCallback(GLMContext glm_ctx);
uint64_t mglRenderEndTimerQueryCallback(GLMContext glm_ctx);
void mglRenderBeginSampleQueryCallback(GLMContext glm_ctx,
                                          unsigned int target);
uint64_t mglRenderEndSampleQueryCallback(GLMContext glm_ctx);

enum {
    MGL_RENDER_AIR_PROGRAM_BOUND = 0,
    MGL_RENDER_AIR_PROGRAM_NOT_APPLICABLE = 1,
    MGL_RENDER_AIR_PROGRAM_ERROR = -1,
};

/* Load every AIR-backed stage in a linked Program and install the resulting
 * +1 library/function references directly in its MGLShaderModule slots.  Programs that
 * still contain a legacy MSL stage are left untouched and return
 * NOT_APPLICABLE so the ObjC baseline can bind the whole program. */
int mglRenderBindAIRProgram(Program *program,
                               int *failed_stage_out,
                               char *err,
                               size_t errcap);

enum {
    MGL_RENDER_BUFFER_BOUND = 0,
    MGL_RENDER_BUFFER_NOT_APPLICABLE = 1,
    MGL_RENDER_BUFFER_ERROR = -1,
};

/* Materialize shared, copy-backed, client-storage, and persistent no-copy
 * Buffer storage in Metal-cpp. No-copy buffers transfer VM-range cleanup to
 * the retained Metal object and set data.mtl_owns_buffer_data. */
int mglRenderBindBufferStorage(Buffer *buffer,
                                  char *err,
                                  size_t errcap);

enum {
    MGL_RENDER_BUFFER_OPERATION_HANDLED = 0,
    MGL_RENDER_BUFFER_OPERATION_NOT_APPLICABLE = 1,
    MGL_RENDER_BUFFER_OPERATION_ERROR = -1,
};

/* Update buffer storage and dirty state for an encoder bind. */
int mglRenderUpdateDirtyBuffer(Buffer *buffer,
                                  char *err,
                                  size_t errcap);
int mglRenderBufferSubDataStorage(Buffer *buffer,
                                     size_t offset,
                                     size_t size,
                                     const void *bytes,
                                     char *err,
                                     size_t errcap);
int mglRenderSnapshotSharedDirtyBuffer(Buffer *buffer,
                                          void **metal_buffer_out,
                                          char *err,
                                          size_t errcap);
int mglRenderSnapshotSharedBufferRange(Buffer *buffer,
                                          size_t offset,
                                          size_t length,
                                          void **metal_buffer_out,
                                          char *err,
                                          size_t errcap);
uint64_t mglRenderAdvanceBufferGeneration(void);
void mglRenderRecordBufferGenerationCompleted(uint64_t generation);
uint64_t mglRenderCompletedBufferGeneration(void);
void mglRenderNoteBufferEncoded(Buffer *buffer);
int mglRenderMapBufferStorage(Buffer *buffer,
                                 size_t offset,
                                 size_t size,
                                 unsigned int access,
                                 bool map,
                                 void **mapped_out,
                                 char *err,
                                 size_t errcap);
int mglRenderFlushBufferRangeStorage(Buffer *buffer,
                                         intptr_t offset,
                                         intptr_t length,
                                         char *err,
                                         size_t errcap);

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

/* Convert unsupported GL vertex formats and return a +1 MTLBuffer as void*.
 * The caller must consume it with __bridge_transfer or release it through
 * mglRenderDeleteMTLObj. The renderer cache owns a separate reference. */
int mglRenderConvertVertexBuffer(
    Buffer *source_buffer,
    const MGLRenderVertexConversion *conversion,
    uint64_t *converted_stride_out,
    void **converted_buffer_out,
    char *err,
    size_t errcap);

/* Pack a plain-struct uniform into renderer-owned transient storage. The
 * returned Buffer wrapper remains owned by the renderer; its MTLBuffer
 * backing is replaced when the 128-slot ring wraps. */
Buffer *mglRenderAcquirePackedStructBuffer(const void *data,
                                               size_t size,
                                               char *err,
                                               size_t errcap);

/* Renderer-owned device utility facade. Newly created resources are +1 and
 * must be consumed by __bridge_transfer or released through the C++ facade. */
int mglRenderCreateBuffer(uint64_t length,
                             uint64_t resource_options,
                             const char *label,
                             void **buffer_out);
int mglRenderCreateBufferWithBytes(const void *bytes,
                                      uint64_t length,
                                      uint64_t resource_options,
                                      const char *label,
                                      void **buffer_out);
/* C++-owned transient upload buffer. The returned buffer is borrowed from
 * the opaque owner; Metal command encoders retain it when a copy command is
 * recorded, so the owner may be destroyed immediately after encoding. */
int mglRenderCreateTextureStagingOwner(
    const void *bytes,
    uint64_t length,
    uint64_t resource_options,
    void **owner_out,
    void **buffer_out);
void mglRenderDestroyTextureStagingOwner(void **owner);
/* Create a shared/no-copy buffer for VM-backed GL client or persistent
 * storage.  When deallocate_vm is non-zero Metal owns the VM range and
 * releases it with vm_deallocate after the last in-flight command buffer. */
int mglRenderCreateBufferWithBytesNoCopy(const void *bytes,
                                            uint64_t length,
                                            uint64_t resource_options,
                                            const char *label,
                                            int deallocate_vm,
                                            void **buffer_out);
int mglRenderGetBufferContents(void *buffer,
                                  void **contents_out,
                                  uint64_t *length_out);
typedef struct MGLRenderBufferInfo_t {
    uint64_t length;
} MGLRenderBufferInfo;
int mglRenderGetBufferInfo(const void *buffer,
                              MGLRenderBufferInfo *info_out);
int mglRenderAddBufferDebugMarker(void *buffer,
                                     const char *marker,
                                     uint64_t location,
                                     uint64_t length);
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
int mglRenderGetTextureInfo(const void *texture,
                               MGLRenderTextureInfo *info_out);
int mglRenderTextureIsFramebufferOnly(const void *texture);
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

/* C++ owns and releases the temporary MTL::TextureDescriptor. The C ABI
 * carries only descriptor values, never an Objective-C/MTL descriptor. */
int mglRenderCreateTextureFromState(
    const MGLRenderTextureDescriptorState *texture_descriptor,
    const char *label,
    void **texture_out);
/* The descriptor is an opaque borrowed Objective-C object. C++ reads its
 * value fields and owns the temporary Metal-cpp descriptor it creates. */
int mglRenderCreateTextureFromDescriptor(
    void *descriptor,
    const char *label,
    void **texture_out);
int mglRenderCreateBufferTextureFromState(
    void *buffer,
    const MGLRenderTextureDescriptorState *texture_descriptor,
    uint64_t offset,
    uint64_t bytes_per_row,
    void **texture_out);
int mglRenderCreateBufferTextureFromDescriptor(
    void *buffer,
    void *descriptor,
    uint64_t offset,
    uint64_t bytes_per_row,
    void **texture_out);
int mglRenderCreateTextureView(void *texture,
                                  uint32_t pixel_format,
                                  void **texture_view_out);
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
/* Apply GL BASE_LEVEL/MAX_LEVEL and swizzle state to a sampled texture. The
 * returned texture is +1 retained for the caller; the Texture cache keeps its
 * own reference. */
int mglRenderSampledTextureViewForBaseLevel(
    Texture *texture_object,
    void *source_texture,
    void **view_out);
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

int mglRenderTextureTargetPlan(
    uint32_t gl_target,
    uint32_t sample_count,
    MGLRenderTextureTargetPlan *plan_out);

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

/* reflected shader-resource image shape ->
 * MTLTextureType ABI value.  The C ABI stays backend-neutral: all inputs and
 * the result are uint32_t values, and has_resource preserves the historical
 * NULL-resource result.  Unsupported dimensions return 0. */
uint32_t mglRenderTextureTypeForShaderResource(
    uint32_t has_resource,
    uint32_t image_dim,
    uint32_t image_arrayed,
    uint32_t image_multisampled);

/* MTLTextureType ABI value -> per-target OpenGL
 * texture-unit slot. Unsupported Metal texture types return -1. */
int32_t mglRenderTextureIndexForMetalType(uint32_t texture_type);

/* MGLPixelFormat ABI value -> shader-visible texture
 * data kind.  Keep the C ABI backend-neutral; the numeric results mirror
 * MGLTextureDataKind without exposing that ObjC enum here. */
#define MGL_RENDER_TEXTURE_DATA_KIND_UNKNOWN 0u
#define MGL_RENDER_TEXTURE_DATA_KIND_FLOAT   1u
#define MGL_RENDER_TEXTURE_DATA_KIND_SINT    2u
#define MGL_RENDER_TEXTURE_DATA_KIND_UINT    3u
#define MGL_RENDER_TEXTURE_DATA_KIND_DEPTH   4u

uint32_t mglRenderTextureDataKindForPixelFormat(uint32_t pixel_format);
/* pure pixel-format and GL internal-format predicates.
 * The C ABI carries only stable integer enum values; ObjC compatibility
 * headers remain thin wrappers around these C++ tables. */
int mglRenderMetalPixelFormatIsDepthOrStencil(uint32_t pixel_format);
int mglRenderMetalPixelFormatIsPackedDepthStencil(uint32_t pixel_format);
int mglRenderGLInternalFormatLooksDepthOrStencil(uint32_t internal_format);
int mglRenderTexturePixelFormatCompatibleWithExpectedDataKind(
    uint32_t pixel_format, uint32_t expected_kind);
/* compressed upload row math.  Returns the block height
 * and rounded upload-row count using uint64_t so the C ABI is Foundation-free. */
uint64_t mglRenderMetalCompressedBlockHeight(uint32_t pixel_format);
uint64_t mglRenderMetalUploadRowsForPixelFormat(uint32_t pixel_format,
                                                   uint64_t pixel_height);
/* data-kind → debug name string (static literals).
 * kind uses MGL_RENDER_TEXTURE_DATA_KIND_*. */
const char *mglRenderTextureDataKindName(uint32_t kind);
/* min-filter → uses-mipmaps.  Returns 1/0. */
int mglRenderTextureMinFilterUsesMipmaps(uint32_t min_filter);

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
int mglRenderPixelFormatIsIntegerColor(uint32_t pixel_format);
int mglRenderPixelFormatIsSignedIntegerColor(uint32_t pixel_format);

/* layer / sRGB pixel-format tables.  Pixel format
 * is the Apple MGLPixelFormat numeric value.  Effective honors
 * GL_EXT_texture_sRGB_decode via the raw srgb_decode_ext enum. */
int mglRenderMetalLayerPixelFormatIsSupported(uint32_t pixel_format);
uint32_t mglRenderSRGBPixelFormat(uint32_t pixel_format);
uint32_t mglRenderLinearPixelFormat(uint32_t pixel_format);
uint32_t mglRenderEffectiveMTLPixelFormat(uint32_t pixel_format,
                                             uint32_t srgb_decode_ext);

/* copy packed rows with optional Y-flip.  Pure CPU
 * memcpy of `row_bytes` per row — mirrors mglMetalCopyRows (void). */
void mglRenderCopyRows(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t row_bytes, uint64_t height, int flip_y);

/* Depth16Unorm / unpacked depth-float rows -> GL
 * float rows with optional Y-flip.  Mirrors the CPU convert loop in
 * mglReadDepthTextureAsFloat (void; bad args are a no-op). */
void mglRenderCopyDepthTextureBytesToFloat(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint64_t src_depth_bytes, int is_depth16, int flip_y);

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

/* copy Metal texture bytes into GL BGRA8 (source-format
 * decode: RGBA8/BGRA8, R/RG/RGBA 8/16/32 unorm/snorm/int/uint/float,
 * RGB9E5, RGB10A2/BGR10A2, BGR5A1, ABGR4, RG11B10, half/float variants)
 * with optional Y-flip.  Pure CPU data transform shared by both gates —
 * mirrors the ObjC mglMetalCopyTextureBytesToBGRA8 exactly (void). */
void mglRenderCopyTextureBytesToBGRA8(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, int flip_y);

/* accepted GL pixel types for
 * mglMetalCopyBGRA8CompatibleTextureBytesToGL.  Returns 1/0. */
int mglRenderReadbackGLTypeAccepted(uint32_t type);

/* SNORM8 texture bytes -> GL format/type, bypassing
 * the lossy BGRA8 UNORM intermediate.  Mirrors the ObjC sourceIsSnorm8
 * path (1 on success, 0 on bad args / unsupported format). */
int mglRenderCopySnorm8TextureBytesToGL(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y);

/* RGB10A2Unorm texture bytes -> GL format/type,
 * bypassing the lossy BGRA8 UNORM intermediate.  Mirrors the ObjC
 * sourceIsRGB10A2Direct path (1 on success, 0 on bad args / unsupported). */
int mglRenderCopyRGB10A2TextureBytesToGL(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y);

/* RG11B10Float texture bytes -> GL format/type,
 * bypassing the lossy BGRA8 UNORM intermediate.  Mirrors the ObjC
 * sourceIsRG11B10FloatDirect path (1 on success, 0 on bad args). */
int mglRenderCopyRG11B10TextureBytesToGL(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y);

/* R16/RG16/RGBA16 Unorm/Snorm/Float and
 * R32/RG32/RGBA32 Float texture bytes -> GL format/type, bypassing
 * the lossy BGRA8 UNORM intermediate.  Mirrors the ObjC 16/32-bit
 * direct path (1 on success, 0 on bad args / unsupported). */
int mglRenderCopy16or32TextureBytesToGL(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y);

/* BGRA8/RGBA8 UNORM texture bytes -> GL scalar
 * types (BYTE/SHORT/INT/UINT/USHORT/HALF/FLOAT).  Mirrors the ObjC
 * scalar integer/half/float readback path (1 on success, 0 on bad
 * args / unsupported). */
int mglRenderCopyUnorm8ScalarTextureBytesToGL(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y);

/* BGRA8/RGBA8 UNORM texture bytes -> GL packed
 * types (3_3_2 / 5_6_5 / 4_4_4_4 / 5_5_5_1 / 8_8_8_8 /
 * 10_10_10_2 / 10F_11F_11F_REV / 5_9_9_9_REV and REV variants).
 * Mirrors the ObjC packed readback path (1 on success, 0 on bad
 * args / unsupported). */
int mglRenderCopyUnorm8PackedTextureBytesToGL(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y);

/* BGRA8/RGBA8 UNORM texture bytes -> GL channel
 * swizzle tail (UNSIGNED_BYTE, plus the leftover RGBA FLOAT branch).
 * Mirrors the ObjC final format switch (1 on success, 0 on bad args /
 * unsupported). */
int mglRenderCopyUnorm8SwizzleTextureBytesToGL(
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

int mglRenderTextureUploadRoute(uint32_t texture_type,
                                   uint32_t storage_mode,
                                   int has_agx_3d_copy_bug);
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
/* Repackages strided 3D depth planes into the tight image stride required by
 * replaceRegion. Returns a malloc-owned buffer or NULL on invalid input or
 * allocation failure. */
void *mglRenderTextureRepackDepthPlanes(const void *bytes,
                                           size_t bytes_per_image,
                                           size_t expected_bytes_per_image,
                                           size_t copy_depth);
/* Expands RGB texels to RGBA for the 2D texel-buffer fallback. The caller owns
 * dst. Missing tail texels are zero-filled and alpha comes from the low
 * dst_comp_bytes of alpha_default. Returns 0 on success. */
int mglRenderTextureExpandRGBToRGBA(const void *src,
                                       void *dst,
                                       size_t texel_count,
                                       size_t tex_width,
                                       size_t tex_height,
                                       size_t src_comp_bytes,
                                       size_t dst_comp_bytes,
                                       uint64_t alpha_default);
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

/* Validate every non-empty entry (bounds vs the Metal buffer lengths) and,
 * when blit_encoder is non-NULL, encode each copy via
 * mglRenderBlitCopyBuffer.  Returns 0 on success, -1 on the first
 * invalid entry / encode failure. */
int mglRenderEncodeStageBindingCopyBacks(
    const MGLRenderCopyBackEntry *entries,
    uint32_t count,
    void *blit_encoder);

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

/* per-level CPU upload data preparation — pure CPU
 * transform shared by both gates (the expansion entries it calls are the
 * same both gates use).  Computes the copy geometry and applies any required
 * format expansion (RGBA8 / channel) to the level bytes.  Returns:
 *   0  success (*out filled; data may be owned — free when owns_data=1)
 *  -1  bad args / rejected level
 *  -2  short backing store (level data smaller than the image needs;
 *      *out still carries the computed geometry for diagnostics) */
typedef struct MGLRenderIntegerReadbackConvertParams_t {
    const uint8_t *src;
    uint64_t src_bytes_per_row;
    uint32_t source_component_count;
    uint32_t source_component_bytes;
    int source_signed;
    int source_rgb10a2_uint;
    uint32_t copy_w;
    uint32_t copy_h;
    uint8_t *dst;
    uint64_t dst_bytes_per_row;
    uint64_t dst_pixel_bytes;
    uint64_t dst_x;
    uint64_t dst_y;
    uint32_t output_components;
    const int *component_map;
    uint32_t output_component_bytes;
    uint32_t packed_type;
    int is_packed_type;
    const uint32_t *packed_bit_widths;
    const uint32_t *packed_shifts;
    uint32_t packed_output_bytes;
} MGLRenderIntegerReadbackConvertParams;

/* integer texture readback CPU conversion — the
 * per-pixel component extraction + GL_INTEGER packing/clamping loop of
 * mglReadIntegerTextureAsRGBA32, as a pure data transformation shared by
 * both gates.  Returns 0 on success, -1 on bad args. */
int mglRenderConvertIntegerReadback(
    const MGLRenderIntegerReadbackConvertParams *params);

/* tess-factor buffer CPU transforms — the default
 * canonical factor fill (12B/patch: 4x outer + 2x inner __fp16), the
 * canonical->triangle repack (12B -> 8B/patch) and the native primitive
 * count (GL 4.6 11.2.2.2 ceil rules).  Pure data transforms shared by both
 * gates.
 * Return 0 on success, -1 on bad args (count entry returns 0). */
int mglRenderFillDefaultTessFactorBuffer(
    void *dst,
    uint64_t dst_bytes,
    const float *outer_levels,
    const float *inner_levels,
    uint32_t patch_count);
int mglRenderRepackTessFactorTriangles(
    const void *src,
    uint64_t src_bytes,
    void *dst,
    uint64_t dst_bytes,
    uint32_t patch_count);
uint64_t mglRenderTessPrimitiveCount(
    const void *factors,
    uint64_t bytes,
    uint32_t patch_count,
    uint32_t tess_gen_mode,
    uint32_t instance_count);

/* GL 4.6 section 11.2.2.2 patch discard predicate.
 * Tests the applicable outer/inner tessellation levels before any clamp to
 * one; non-positive or NaN levels discard the patch.  NULL inputs are
 * conservatively treated as discarded.  Shared by both gates. */
bool mglRenderTessFactorsDiscardPatch(
    uint32_t gen_mode,
    const float *edge,
    const float *inside);

/* per-patch expanded item count for the isolines /
 * point-mode TES kernel (lockstep with mgl_air_backend.cpp's u/v
 * decomposition) — returns 0 when the factor record is missing or the patch
 * is discarded (caller falls back to 1).  Pure data transform shared by
 * both gates. */
uint32_t mglRenderTessEvalItemsPerPatch(
    const void *factor_record,
    uint32_t gen_mode,
    uint32_t spacing,
    uint32_t point_mode);

/* GL 4.6 §11.2.2.2 subdivision-count rounding —
 * fractional_even -> next even (min 2), fractional_odd -> next odd,
 * otherwise ceil(level).  Single source of truth shared by the TES
 * eval-item accounting and the ObjC native per-patch primitive counting
 * (mglTessRoundLevelForSpacing shell in MGLRenderer+Tessellation.m). */
uint32_t mglRenderTessRoundLevelForSpacing(
    uint32_t spacing,
    uint32_t ceil_level);

/* TES XFB field byte size for a GL type (FLOAT/INT/
 * UINT + vec2/3/4; 0 for unsupported).  Matches mglTESXFBFieldByteSize and
 * the packed-write stride contract in mglFixMSLTesAsComputeKernel.  Shared
 * by both gates. */
uint64_t mglRenderTESXFBFieldByteSize(uint64_t gl_type);

/* overflow-checked product (a * b) for tessellation
 * size math; matches the ObjC mglCheckedNSUIntegerProduct.  Returns 0 with
 * *result set, -1 on bad args / overflow.  Shared by both gates. */
int mglRenderCheckedProduct(uint64_t a, uint64_t b, uint64_t *result);

/* unpack an 11-bit (6-bit mantissa) / 10-bit
 * (5-bit mantissa) unsigned float — CPU decode for
 * GL_UNSIGNED_INT_10F_11F_11F_REV vertex data.  5-bit exponent bias 15,
 * no sign bit; matches the ObjC mglFloat11ToFloat / mglFloat10ToFloat
 * exactly (denormal, inf, NaN and ldexpf paths).  Shared by both gates. */
float mglRenderFloat11ToFloat(uint32_t val);
float mglRenderFloat10ToFloat(uint32_t val);

/* CPU pixel-format scalar converters shared by the
 * readback path (mgl_readback.m's mglMetalFloatToUnorm8 /
 * mglMetalSnorm16ToFloat / mglMetalSnorm8ToFloat — pure data transforms,
 * both gates).  Float->unorm8 rounds to nearest (0.5 rounds up); snorm
 * decode maps INT_MIN to -1.0 exactly. */
uint8_t mglRenderFloatToUnorm8(float value);
float mglRenderSnorm16ToFloat(int16_t value);
float mglRenderSnorm8ToFloat(int8_t value);

/* GL type -> MTLVertexFormat ABI value for TES
 * control-point stage inputs (Float/Float2/3/4, Int/Int2/3/4,
 * UInt/UInt2/3/4, else 0 = MTLVertexFormatInvalid).  Values match the
 * macOS SDK enum (Float=28 ... UInt4=39).  Shared by both gates. */
uint32_t mglRenderTessControlPointFormat(uint64_t gl_type);

/* TES XFB compact vertex stride — sum of the byte
 * sizes of the transform-feedback varyings resolved by name against the
 * TES stage-output resource list (lockstep with the packed writes injected
 * by mglFixMSLTesAsComputeKernel).  0 when the stride cannot be proven
 * (no varyings / unknown field type / overflow).  Matches the ObjC
 * mglTESXFBVertexStride.  Shared by both gates. */
uint64_t mglRenderTESXFBVertexStride(const void *program);

/* Overflow-checked tess capture size (records x stride, min_stride floor).
 * Returns 0 with size_out/offset_out set, -1 on bad args / overflow. */
int mglRenderCheckedTessCaptureSize(
    int64_t count,
    int64_t instance_count,
    uint64_t stride,
    uint64_t min_stride,
    uint64_t *size_out,
    uint64_t *offset_out);

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

typedef struct MGLRenderIntegerReadbackClassify_t {
    int source_is_integer_texture;
    int output_is_integer_format;
    uint32_t output_components;
    int component_map[4];
    uint32_t output_component_bytes;
} MGLRenderIntegerReadbackClassify;

/* integer-readback classification — the 19-format
 * source-integer table, the GL_*_INTEGER output check, the per-format
 * component map (incl. BGR/BGRA orderings and the GREEN/BLUE/ALPHA
 * single-component compat enums) and the per-type output component bytes.
 * Pure classification shared by both gates.  Returns 0 on success, -1 on
 * bad args. */
int mglRenderIntegerReadbackClassify(
    uint32_t pixel_format,
    uint32_t gl_format,
    uint32_t gl_type,
    MGLRenderIntegerReadbackClassify *out);

typedef struct MGLRenderIntegerPackedType_t {
    int is_packed;
    uint32_t bit_widths[4];
    uint32_t shifts[4];
    uint32_t output_bytes;
    uint32_t output_components;
} MGLRenderIntegerPackedType;

/* integer-readback packed-type classification —
 * the 10-entry GL packed-type table (3_3_2 / 2_3_3_REV / 5_6_5(+REV) /
 * 4_4_4_4(+REV) / 5_5_5_1 / 1_5_5_5_REV / 8_8_8_8(+REV) /
 * 10_10_10_2 / 2_10_10_10_REV).  Pure classification shared by both
 * gates.  Returns 0 on success, -1 on bad args. */
int mglRenderIntegerReadbackPackedTypeClassify(
    uint32_t packed_type,
    MGLRenderIntegerPackedType *out);

typedef struct MGLRenderIntegerReadbackSource_t {
    uint32_t component_count;
    uint32_t component_bytes;
    int source_signed;
    int source_rgb10a2_uint;
    int recognized;
} MGLRenderIntegerReadbackSource;

/* integer-readback SOURCE format classification —
 * the 19-entry MGLPixelFormat -> {components, component bytes, signed,
 * RGB10A2} table.  Pure classification shared by both gates.  Returns 0
 * with recognized=1 on a known format, 0 with recognized=0 on unknown,
 * -1 on bad args. */
int mglRenderIntegerReadbackSourceClassify(
    uint32_t pixel_format,
    MGLRenderIntegerReadbackSource *out);

/* shadow-upload range math — for gpu_write_target
 * buffers, clamps the recorded written_min/written_max span to the limit;
 * otherwise the whole limit.  Returns 0 with offset/length set, -1 when
 * there is nothing to upload (no written span / zero length).  Pure range
 * computation shared by both gates. */
int mglRenderBufferShadowUploadRange(
    int gpu_write_target,
    int64_t written_min,
    int64_t written_max,
    uint64_t limit,
    uint64_t *out_offset,
    uint64_t *out_length);

/* GL draw mode -> MTLPrimitiveType numbering
 * (0=Point, 1=Line, 2=LineStrip, 3=Triangle, 4=TriangleStrip;
 * 0xFFFFFFFF for modes the renderer routes elsewhere).  Pure table shared
 * by both gates; the caller casts to MTLPrimitiveType. */
uint32_t mglRenderMTLPrimitiveTypeForGLMode(uint32_t mode);

/* GL element index type -> MTLIndexType numbering
 * (0=UInt16, 1=UInt32; 0xFFFFFFFF otherwise).  Pure table shared by both
 * gates; the caller casts to MTLIndexType. */
uint32_t mglRenderMTLIndexTypeForGLType(uint32_t gl_type);

/* Metal mipmap level dimension — the greatest
 * 2^(level) divisor of base (base>>level, clamped to 1).  Pure computation
 * shared by both gates (the ObjC mglMetalTextureLevelDimension keeps the
 * extern linkage its many callers use). */
uint64_t mglRenderMetalTextureLevelDimension(uint64_t base, uint64_t level);

/* triangle-fan element emulation — expand a raw
 * element index stream into `(center, i+1, i+2)` triplets (count-2
 * triangles x 3, all uint32).  Pure CPU; caller frees the returned array.
 * Returns 0 on success with *out_count set, -1 on bad args / overflow. */
int mglRenderExpandTriangleFanIndices(
    const uint8_t *bytes,
    uint32_t elem_width,        /* 1, 2 or 4 */
    uint32_t source_count,
    uint32_t **out_indices,     /* malloc'd, count*3 entries */
    uint64_t *out_count);

/* triangle-strip element emulation — expand a raw
 * element stream into `(first, second, tri+2)` triplets with alternating
 * first/second offset (tri strips), count-2 triangles, all uint32.
 * Pure CPU; caller frees.  Returns 0 with *out_count set, -1 on error. */
int mglRenderExpandTriangleStripIndices(
    const uint8_t *bytes, uint32_t elem_width, uint32_t source_count,
    uint32_t **out_indices, uint64_t *out_count);

/* LINE_LOOP element emulation — copy the raw index
 * stream and append the first index to close the loop (count+1).  Pure CPU;
 * caller frees. */
int mglRenderExpandLineLoopIndices(
    const uint8_t *bytes, uint32_t elem_width, uint32_t source_count,
    uint32_t **out_indices, uint64_t *out_count);

/* quad-array emulation — for each group of 4 array
 * vertices emit `(a,a+1,a+2,a,a+2,a+3)` (two triangles), quad_count*6
 * uint32 total.  Pure CPU; caller frees.  Returns 0 with *out_count, -1 on
 * bad args. */
int mglRenderExpandQuadArrayIndices(
    uint32_t quad_count, uint32_t **out_indices, uint64_t *out_count);

/* quad-element emulation — read 4 source indexes per
 * quad from the raw stream and emit `(i0,i1,i2,i0,i2,i3)`.  Pure CPU;
 * caller frees. */
int mglRenderExpandQuadElementIndices(
    const uint8_t *bytes, uint32_t elem_width, uint32_t quad_count,
    uint32_t **out_indices, uint64_t *out_count);

/* GL_UNSIGNED_BYTE element buffer -> UInt16
 * expansion — write each byte as uint16.  Pure CPU; caller frees. */
int mglRenderExpandUInt8ToUInt16(
    const uint8_t *bytes, uint32_t byte_count,
    uint16_t **out_indices, uint64_t *out_count);

/* triangle-fan ARRAY emulation — vertexCount-2
 * triangles `(0, tri+1, tri+2)`, all uint32.  Pure CPU; caller frees. */
int mglRenderExpandTriangleFanArrayIndices(
    uint32_t vertex_count, uint32_t **out_indices, uint64_t *out_count);

/* triangle-strip ARRAY emulation — vertexCount-2
 * triangles with alternating offset `(tri&1)`.  Pure CPU; caller frees. */
int mglRenderExpandTriangleStripArrayIndices(
    uint32_t vertex_count, uint32_t **out_indices, uint64_t *out_count);

/* LINE_LOOP ARRAY emulation — copy `firstVertex+i`
 * for count vertices then append `firstVertex`.  Pure CPU; caller frees. */
int mglRenderExpandLineLoopArrayIndices(
    uint32_t first_vertex, uint32_t vertex_count,
    uint32_t **out_indices, uint64_t *out_count);

/*  (: quad-array LINE_LOOP emulation — for each group of
 * 4 array vertices emit `(a,a+1,a+1,a+2,a+2,a+3,a+3,a)` (a 4-edge closed
 * loop), quad_count*8 uint32 total.  Pure CPU; caller frees. */
int mglRenderExpandQuadArrayLineIndices(
    uint32_t quad_count, uint32_t **out_indices, uint64_t *out_count);

/* quad-element LINE_LOOP emulation — read 4 source
 * indexes per quad and emit `(i0,i1,i1,i2,i2,i3,i3,i0)`.  Pure CPU;
 * caller frees. */
int mglRenderExpandQuadElementLineIndices(
    const uint8_t *bytes, uint32_t elem_width, uint32_t quad_count,
    uint32_t **out_indices, uint64_t *out_count);

/* index-range scan ignoring primitive-restart markers
 * — computes min/max over the byte stream (BYTE/SHORT/INT width), skipping
 * the restart value.  Pure CPU; matches mglScanIndexRangeIgnoringRestart.
 * Returns 0 on success (with *out_valid = 1 if at least one non-restart
 * index was seen), -1 on bad args. */
int mglRenderScanIndexRangeIgnoringRestart(
    const uint8_t *bytes, uint32_t elem_width, uint32_t count,
    int restart_enabled, uint32_t restart_index,
    uint32_t *out_min, uint32_t *out_max, int *out_valid);

/* prepared (Metal-side) byte offset for a GL element
 * buffer — GL_UNSIGNED_BYTE indices are expanded to UInt16 so the offset
 * doubles, other types pass through.  Matches mglComputePreparedIndexByteOffset.
 * Returns 0 on success, -1 on overflow / bad args. */
int mglRenderComputePreparedIndexByteOffset(uint64_t gl_index_type,
                                               uint64_t gl_byte_offset,
                                               uint64_t *out_prepared_offset);

/* baseByteOffset + firstElement * indexStride with
 * overflow checks.  Matches mglComputeIndexByteOffset.  Returns 0 on success,
 * -1 on bad args / overflow. */
int mglRenderComputeIndexByteOffset(uint64_t base_byte_offset,
                                       uint64_t first_element,
                                       uint64_t index_stride,
                                       uint64_t *out_byte_offset);

/* GL index element byte size (BYTE=1, SHORT=2, INT=4).
 * Matches mglGLIndexElementSize.  Returns 0 for unknown type. */
uint32_t mglRenderGLIndexElementSize(uint64_t gl_index_type);

/* read a single GL index value from a byte buffer at
 * `element_index` (elem_width 1/2/4).  Matches mglReadGLIndexValue; returns 0
 * for NULL buffer or unknown width. */
uint32_t mglRenderReadGLIndexValue(const uint8_t *bytes, uint32_t elem_width,
                                      uint64_t element_index);

/* GL vertex-attribute component size in bytes (1/2/4/8).
 * Matches mglVertexAttribComponentSize.  Returns 0 for unknown. */
uint32_t mglRenderVertexAttribComponentSize(uint64_t gl_type);

/* total bytes for a vertex-attribute element (type x
 * size), with special handling for packed 10_10_10_2 formats.  Matches
 * mglVertexAttribElementBytes.  Returns 0 for unknown / zero size. */
uint64_t mglRenderVertexAttribElementBytes(uint64_t gl_type, uint32_t size);

/* does GL primitive mode produce polygonal primitives
 * (triangles/quads) subject to glPolygonMode point/line emulation?  Matches
 * mglDrawModeProducesPolygons.  Returns 1/0. */
int mglRenderDrawModeProducesPolygons(uint64_t gl_mode);

/* does `mode` with `indexCount` vertices produce at
 * least one drawable segment (point/line/triangle/quad)?  Matches
 * mglPrimitiveModeHasDrawableSegment.  Returns 1/0. */
int mglRenderPrimitiveModeHasDrawableSegment(uint64_t gl_mode,
                                                uint64_t index_count);

/* total triangle index count for `source_vertex_count`
 * vertices arranged as quads (4/quad -> 6 indices).  Matches
 * mglQuadTriangleIndexCount; returns 0 on overflow. */
uint64_t mglRenderQuadTriangleIndexCount(uint64_t source_vertex_count);
/* Align vertex stride to 4; matches mglAlignVertexStrideForMetal. */
uint64_t mglRenderAlignVertexStrideForMetal(uint64_t stride);
/* double-attrib size -> MTLVertexFormat value; matches mglDoubleVertexAttribFloatFormat. */
uint32_t mglRenderDoubleVertexAttribFloatFormat(uint32_t size);
/* Integer attrib signedness mismatch -> Int/UInt MTLVertexFormat ABI value.
 * Returns MTLVertexFormatInvalid when no CPU conversion is required. */
uint32_t mglRenderIntegerAttribConversionFormat(
    uint64_t src_type,
    uint64_t shader_gl_type,
    uint32_t size);
const char *mglRenderVertexFormatName(uint32_t format);
uint64_t mglRenderVertexDescriptorSignature(const void *descriptor);
uint64_t mglRenderPipelineDescriptorSignature(const void *descriptor);
/* FNV-1a single hash step; matches mglHashStepU64. */
uint64_t mglRenderHashStepU64(uint64_t hash, uint64_t value);
/* Fixed restart-index for a type; matches the fixed branch of
 * mglPrimitiveRestartIndexForType.  1 if defined; *out set. */
int mglRenderPrimitiveRestartFixedIndex(uint64_t gl_index_type, uint32_t *out);
/* GL uniform/attrib type -> element byte size; matches mglGLTypeElementByteSize. */
uint32_t mglRenderGLTypeElementByteSize(uint64_t gl_type);

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

/* readPixels region-vs-level clip — clamps a source
 * read region against the level extents and computes the destination
 * offset-origin for the clipped copy and the Metal source Y (flipped).
 * Pure computation shared by both gates; the empty flag matches the
 * original `copyW <= 0 || copyH <= 0`. */
int mglRenderReadTextureRegionClip(
    int64_t region_x, int64_t region_y,
    int64_t region_w, int64_t region_h,
    int64_t level_w, int64_t level_h,
    MGLRenderReadTextureRegionClip *out);

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

/* ARB_vertex_attrib_binding resolve — the
 * binding-table override (offset/stride/divisor) vs the legacy per-attrib
 * values.  Pure decision shared by both gates; the GL buffer validation
 * stays on the ObjC side. */
int mglRenderResolveVertexAttribBinding(
    uint32_t binding_index,
    int binding_has_buffer,
    int64_t binding_offset,
    uint32_t binding_stride,
    int64_t attrib_binding_offset,
    uint32_t attrib_stride,
    uint32_t binding_divisor,
    uint32_t attrib_divisor,
    MGLRenderVertexAttribResolve *out);

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

/* GL draw mode -> primitive vertex count (for the
 * cull-distance emulation params; 1 for unknown modes).  Pure table shared
 * by both gates. */
uint32_t mglRenderPrimitiveVertexCountForMode(uint32_t mode);

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

/* scaled-blit UV computation (normalized source
 * rect with the Metal Y-flip, clamped, direction-swapped per the forward
 * flags).  Pure CPU, shared by both gates. */
int mglRenderScaledBlitUVs(
    uint32_t src_tex_w,
    uint32_t src_tex_h,
    double src_min_x,
    double src_max_x,
    double src_min_y,
    double src_max_y,
    int src_x_forward,
    int src_y_forward,
    int dst_x_forward,
    int dst_y_forward,
    MGLRenderScaledBlitUVs *out);

/* scaled-blit destination scissor base — floor/ceil
 * of the destination rect in Metal Y, clamped to the destination texture.
 * The caller intersects the GL scissor box on top.  Pure CPU, shared by
 * both gates. */
int mglRenderBlitScissorRect(
    double dst_min_x,
    double dst_max_x,
    double scaled_dst_metal_y,
    double dst_h,
    uint32_t dst_tex_w,
    uint32_t dst_tex_h,
    MGLRenderBlitScissorRect *out);

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

/* glBlitFramebuffer region math + decisions after
 * the axis clip — direction/flip flags, min/max/abs extents, the scaled-
 * blit decision (format conversion / RT sync / scissor / flip / size
 * mismatch with the 1e-5 epsilon of mglNearlyEqual), the integer copy
 * rect, the Metal Y-flips and the scaled-path destination Y.  Pure CPU
 * plan shared by both gates.  Returns 0 with the plan filled, -1 when the
 * clipped region has zero extent (caller logs and skips). */
int mglRenderBlitFramebufferPlan(
    double src_x0,
    double src_x1,
    double src_y0,
    double src_y1,
    double dst_x0,
    double dst_x1,
    double dst_y0,
    double dst_y1,
    uint32_t src_tex_w,
    uint32_t src_tex_h,
    uint32_t dst_tex_w,
    uint32_t dst_tex_h,
    int needs_format_conversion_blit,
    int needs_render_target_sync_blit,
    int scissor_test_enabled,
    MGLRenderBlitFramebufferPlan *out);

typedef struct MGLRenderGetTexImagePlan_t {
    int direct_r32_float_read;
    int use_bgra8_conversion;
    int source_is_bgra8;
    uint64_t row_bytes;
    uint64_t image_bytes;
    uint64_t total_bytes;
} MGLRenderGetTexImagePlan;

/* mtlGetTexImage staging plan — direct R32F read
 * detection, the BGRA8 conversion eligibility (dst bytes + single depth
 * layer + non-direct + compatible source), the source-is-BGRA8-family
 * check, and the row/image/total byte computation (conversion pitch:
 * width*sourceBpp for non-BGRA8 sources, width*4 for BGRA8 sources;
 * otherwise bytesPerRow or width*max(dst,1); the depth>1 + bytesPerImage
 * case applies to private storage only).  Shared by both gates; the caller
 * resolves sizeForFormatType / readback bytes-per-pixel / format
 * compatibility through the existing C helpers. */
int mglRenderGetTexImagePlan(
    uint32_t pixel_format,
    uint32_t gl_format,
    uint32_t gl_type,
    uint32_t width,
    uint32_t height,
    uint32_t depth,
    uint32_t dst_pixel_bytes,
    uint32_t source_bpp,
    int bgra8_format_compatible,
    uint32_t bytes_per_row,
    uint32_t bytes_per_image,
    int storage_private,
    MGLRenderGetTexImagePlan *out);

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

int mglRenderTexturePrepareLevelUpload(
    const TextureLevel *level,
    uint32_t texture_type,
    uint32_t internal_format,
    uint32_t pixel_format,
    MGLRenderLevelUploadPrep *out);

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
/* RGB-family → RGBA expansion gates.  Pixel format is
 * the Apple MGLPixelFormat numeric value.  Returns 1/0. */
int mglRenderTextureInternalFormatNeedsRGBA8Expansion(
    uint32_t internal_format, uint32_t pixel_format);
int mglRenderTextureNeedsChannelExpansion(uint32_t internal_format,
                                             uint32_t pixel_format);

/* R8 swizzle component + single-channel upload expand.
 * Resolve mirrors mglResolveR8SwizzledComponent (tex unused).  Create
 * expands GL_R8 1B/px → RGBA8 via the four swizzle enums; malloc'd
 * result, NULL on bad args / non-R8 / size cap. */
uint8_t mglRenderResolveR8SwizzledComponent(uint32_t swizzle, uint8_t red);
/* R-only upload-swizzle gate.  swizzled==0 → 0;
 * otherwise the GL_R* internal-format table.  Returns 1/0. */
int mglRenderTextureUploadNeedsSingleChannelSwizzle(uint32_t internal_format,
                                                       int swizzled);
/* Metal pixel format for single-channel swizzle upload expansion.
 * Returns MTLPixelFormatInvalid when the format is not handled. */
uint32_t mglRenderSingleChannelSwizzleStoragePixelFormat(
    uint32_t internal_format);
/* Multi-channel integer formats bake swizzle into CPU texels instead of
 * relying on Metal view swizzle (unreliable for Sint on some paths). */
int mglRenderTextureUploadNeedsIntegerMultiChannelSwizzleBake(
    uint32_t internal_format, int swizzled);
/* Returns 1 when swizzle was baked at upload for this storage format. */
int mglRenderTextureSwizzleUsesUploadBake(
    uint32_t internal_format, int swizzled,
    uint32_t storage_pixel_format);
/* stored color-component count for an internal format.
 * Mirrors mglStoredColorComponentsForTexture after the null-tex check
 * (null stays in ObjC and returns 4).  Unknown formats → 4. */
uint32_t mglRenderStoredColorComponents(uint32_t internal_format);
/* GL swizzle enum → Metal TextureSwizzle ABI value
 * (uint32_t).  components gates missing channels to Zero / One(for Alpha). */
uint32_t mglRenderMTLSwizzleForGLSwizzle(uint32_t gl_swizzle,
                                            uint32_t components);
uint8_t *mglRenderCreateSingleChannelSwizzledUpload(
    uint32_t internal_format,
    uint32_t swizzle_r, uint32_t swizzle_g,
    uint32_t swizzle_b, uint32_t swizzle_a,
    const void *src_data, size_t width, size_t height,
    size_t src_bytes_per_row,
    size_t *out_bytes_per_row, size_t *out_bytes_per_image);
uint8_t *mglRenderCreateIntegerMultiChannelSwizzledUpload(
    uint32_t internal_format,
    uint32_t swizzle_r, uint32_t swizzle_g,
    uint32_t swizzle_b, uint32_t swizzle_a,
    const void *src_data, size_t width, size_t height,
    size_t src_bytes_per_row,
    size_t *out_bytes_per_row, size_t *out_bytes_per_image);
int mglRenderCreateSampler(void *sampler_descriptor,
                              void **sampler_out);
int mglRenderCreateDefaultSampler(void **sampler_out);
int mglRenderCreateFilterSampler(uint32_t nearest, void **sampler_out);
/* Translate GL texture parameters into a Metal-cpp sampler descriptor and
 * create the sampler without exposing MTL::* through this C ABI. */
int mglRenderCreateSamplerForGL(const TextureParameter *params,
                                   uint32_t target,
                                   void **sampler_out,
                                   char *err,
                                   size_t errcap);
int mglRenderCreateDepthStencilState(void *depth_stencil_descriptor,
                                        void **depth_stencil_state_out);

enum {
    MGL_RENDER_PIPELINE_CACHE_KEY_WORDS = 7,
    MGL_RENDER_PIPELINE_COLOR_ATTACHMENTS = 8,
};

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

/* Return stable device identity data for platform-neutral cache naming. */
int mglRenderGetDeviceIdentity(const void *device,
                                  uint64_t *registry_id_out,
                                  char *name_out,
                                  size_t name_capacity);

int mglRenderCreateDepthStencilStateFromState(
    const MGLRenderDepthStencilDescriptorState *descriptor,
    void **depth_stencil_state_out);

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

/* Per-renderer pipeline ownership. The opaque owner retains active objects,
 * cached PSOs/functions/descriptors, and depth-stencil states. All returned
 * object pointers are borrowed for the lifetime of the owner/cache entry. */
int mglRenderCreatePipelineCacheOwner(
    int pso_dedup_enabled,
    int depth_stencil_cache_enabled,
    int binary_archive_enabled,
    void **owner_out);
void mglRenderDestroyPipelineCacheOwner(void **owner);
void mglRenderResetPipelineCacheOwner(void *owner);
int mglRenderGetPipelineCacheFlags(
    void *owner,
    int *pso_dedup_enabled_out,
    int *depth_stencil_cache_enabled_out,
    int *binary_archive_enabled_out);
void mglRenderDisablePipelineBinaryArchive(void *owner);
int mglRenderGetPipelineBinaryArchiveState(
    void *owner, int *enabled_out, int *present_out);
int mglRenderLoadPipelineBinaryArchive(
    void *owner,
    const char *cache_key,
    void *url,
    int archive_exists,
    int *reused_out,
    char *err,
    size_t errcap);
int mglRenderSerializePipelineBinaryArchive(
    void *owner, void *url, char *err, size_t errcap);
void mglRenderDiscardPipelineBinaryArchive(
    void *owner, const char *cache_key);
int mglRenderGetPipelineActiveState(
    void *owner, MGLRenderPipelineActiveState *state_out);
int mglRenderInvalidatePipelineActiveState(void *owner);
int mglRenderSetPipelineActiveObject(void *owner, void *pipeline_state);
int mglRenderActivatePipelineState(
    void *owner, const MGLRenderPipelineActiveState *state);
int mglRenderSetPipelineBlendState(
    void *owner, uint32_t attachment,
    const MGLRenderPipelineBlendState *state);
int mglRenderGetPipelineBlendState(
    void *owner, uint32_t attachment,
    MGLRenderPipelineBlendState *state_out);
int mglRenderGetOrCreateDepthStencilState(
    void *owner,
    const MGLRenderDepthStencilDescriptorState *descriptor,
    void **depth_stencil_state_out,
    int *created_out);
int mglRenderLookupPipeline(
    void *owner,
    const uint64_t key_words[MGL_RENDER_PIPELINE_CACHE_KEY_WORDS],
    MGLRenderPipelineActiveState *state_out);
int mglRenderStorePipeline(
    void *owner,
    const uint64_t key_words[MGL_RENDER_PIPELINE_CACHE_KEY_WORDS],
    const MGLRenderPipelineActiveState *state,
    uint32_t *evicted_out);
/* Value-state descriptor cache. A hit returns the complete descriptor state. */
int mglRenderLookupPipelineDescriptorState(
    void *owner,
    const uint64_t key_words[MGL_RENDER_PIPELINE_CACHE_KEY_WORDS],
    MGLRenderPipelineDescriptorState *state_out);
int mglRenderStorePipelineDescriptorState(
    void *owner,
    const uint64_t key_words[MGL_RENDER_PIPELINE_CACHE_KEY_WORDS],
    const MGLRenderPipelineDescriptorState *state);
int mglRenderCreateEvent(void **event_out);
int mglRenderCreateFunction(void *library,
                               const char *name,
                               void *function_constant_values,
                               void **function_out,
                               char *err,
                               size_t errcap);
int mglRenderCreateRenderPipelineState(
    void *render_pipeline_descriptor,
    void **pipeline_out,
    char *err,
    size_t errcap);
/* Descriptor-based archive-aware creation used by the temporary ObjC
 * descriptor paths. Complete VS+FS pipelines query the archive first and are
 * added only on a miss. archive_hit_out is optional and receives 1 only when
 * the returned PSO came directly from the archive. */
int mglRenderCreateRenderPipelineStateWithArchive(
    void *render_pipeline_descriptor,
    void *binary_archive,
    void **pipeline_out,
    int *archive_hit_out,
    char *err,
    size_t errcap);
int mglRenderCreateRenderPipelineStateWithArchiveOwner(
    void *owner,
    void *render_pipeline_descriptor,
    void **pipeline_out,
    int *archive_hit_out,
    char *err,
    size_t errcap);
/* Creates a render PSO from value-state. Function and binary-archive pointers
 * are borrowed; binary_archive may be NULL. On success pipeline_out receives
 * an owned reference that must be released with mglAirRelease. */
int mglRenderCreateRenderPipelineFromState(
    void *vs_function,
    void *fs_function,
    const MGLRenderPipelineDescriptorState *state,
    void *binary_archive,
    void **pipeline_out,
    char *err,
    size_t errcap);
int mglRenderCreateRenderPipelineFromStateWithArchiveOwner(
    void *owner,
    void *vs_function,
    void *fs_function,
    const MGLRenderPipelineDescriptorState *state,
    void **pipeline_out,
    char *err,
    size_t errcap);
int mglRenderCreateComputePipelineState(void *function,
                                           void **pipeline_out,
                                           char *err,
                                           size_t errcap);
uint32_t mglRenderComputePipelineMaxTotalThreads(void *pipeline);
int mglRenderCreateBinaryArchive(void *binary_archive_descriptor,
                                    const char *label,
                                    void **binary_archive_out,
                                    char *err,
                                    size_t errcap);
int mglRenderSerializeBinaryArchive(void *binary_archive,
                                       void *url,
                                       char *err,
                                       size_t errcap);
int mglRenderSetVisibilityResultMode(void *render_encoder,
                                        uint32_t mode,
                                        uint64_t offset);
int mglRenderSetVisibilityResultModeForRenderEncoderOwner(
    void *render_encoder_owner,
    uint32_t mode,
    uint64_t offset);
int mglRenderSampleTimestamps(uint64_t *cpu_timestamp_out,
                                 uint64_t *gpu_timestamp_out);
int mglRenderCreateQueryStateOwner(uint32_t visibility_slot_count,
                                      void **owner_out);
int mglRenderBeginSampleQuery(void *owner,
                                 uint32_t counting,
                                 const char *buffer_label,
                                 void **visibility_buffer_out);
int mglRenderGetQueryVisibilityBuffer(void *owner,
                                         void **visibility_buffer_out);
void mglRenderEndSampleQuery(void *owner);
int mglRenderIsSampleQueryActive(void *owner, uint32_t *active_out);
int mglRenderAcquireSampleQuerySlot(void *owner,
                                       uint32_t *mode_out,
                                       uint64_t *offset_out);
int mglRenderGetSampleQueryResult(void *owner, uint64_t *result_out);
int mglRenderBeginTimerQuery(void *owner);
int mglRenderEndTimerQuery(void *owner, uint64_t *elapsed_out);
void mglRenderDestroyQueryStateOwner(void **owner);

/* Create or reuse a compute PSO owned by the C++ renderer. function is the
 * actual MTLFunction selected by the caller, preserving AIR stage variants.
 * On success *pipeline_out is a +1 MTLComputePipelineState reference that the
 * ObjC bridge may consume with __bridge_transfer. */
int mglRenderGetOrCreateComputePipeline(
    void *function,
    uint64_t program_instance,
    uint64_t program_generation,
    uint32_t stage,
    int cache_enabled,
    void **pipeline_out,
    char *err,
    size_t errcap);

/* Drop all C++ compute PSOs for a Program lifetime. Called before relink and
 * final Program destruction; safe before renderer initialization. */
void mglRenderInvalidateProgramPipelines(uint64_t program_instance);

enum {
    MGL_RENDER_AUX_COMPUTE_SCALED_BLIT = 1,
    MGL_RENDER_AUX_COMPUTE_MSAA_INTEGER_RESOLVE = 2,
    MGL_RENDER_AUX_RENDER_SCALED_BLIT = 3,
    MGL_RENDER_AUX_RENDER_SCALED_DEPTH_BLIT = 4,
    MGL_RENDER_AUX_RENDER_CLEAR_RECT = 5,
    MGL_RENDER_AUX_COMPUTE_GS_XFB_SCATTER = 6,
};

/* Lookup or create a renderer-lifetime auxiliary compute PSO. Passing a NULL
 * function performs lookup only and returns 1 on a cache miss. On success the
 * returned pipeline is an independent +1 reference. */
int mglRenderGetOrCreateAuxComputePipeline(
    void *function,
    uint32_t kind,
    uint64_t variant,
    void **pipeline_out,
    char *err,
    size_t errcap);

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

/* Aux compute PSO from the precompiled aux shader asset table. entry_name is
 * the metallib kernel name. On success *pipeline_out is a +1
 * MTL::ComputePipelineState reference. */
int mglRenderGetOrCreateAuxComputePipelineFromMetallib(
    const unsigned char *bytes,
    size_t size,
    uint64_t asset_hash,
    const char *entry_name,
    uint32_t kind,
    uint64_t variant,
    void **pipeline_out,
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

/* Per-renderer-context binding dedup state. Metal objects stored in this
 * handle are retained by C++ and released on replacement, invalidation, or
 * destroy. Setter calls return 1 when encoded, 0 when deduplicated, and -1
 * for invalid arguments. */
void *mglRenderBindingCreate(uint32_t max_texture_slots);
void mglRenderBindingDestroy(void *binding_state);
void mglRenderBindingInvalidate(void *binding_state);
void mglRenderBindingSetValid(void *binding_state, int valid);
int mglRenderBindingGetValid(void *binding_state, uint32_t *valid_out);
int mglRenderBindingGetTextureSlotMask(void *binding_state,
                                          uint64_t mask_out[2]);
int mglRenderBindingRecordVertexBuffer(void *binding_state,
                                          void *buffer,
                                          uint64_t offset,
                                          uint32_t index);
int mglRenderBindingRecordFragmentBuffer(void *binding_state,
                                            void *buffer,
                                            uint64_t offset,
                                            uint32_t index);
int mglRenderBindingInvalidateVertexBuffer(void *binding_state,
                                              uint32_t index);
int mglRenderBindingInvalidateFragmentBuffer(void *binding_state,
                                                uint32_t index);
int mglRenderBindingUpdateVertexBuffer(void *binding_state,
                                          void *buffer,
                                          uint64_t offset,
                                          uint32_t index);
int mglRenderBindingUpdateFragmentBuffer(void *binding_state,
                                            void *buffer,
                                            uint64_t offset,
                                            uint32_t index);
int mglRenderBindingClearVertexBuffer(void *binding_state,
                                         uint32_t index);
int mglRenderBindingClearFragmentBuffer(void *binding_state,
                                           uint32_t index);
int mglRenderBindingGetBuffer(void *binding_state,
                                 uint32_t stage,
                                 uint32_t index,
                                 void **buffer_out,
                                 uint64_t *offset_out);
void mglRenderBindingOrVertexBufferMask(void *binding_state,
                                           uint32_t mask);
void mglRenderBindingOrFragmentBufferMask(void *binding_state,
                                             uint32_t mask);
void mglRenderBindingSetPipelineState(void *binding_state,
                                         void *pipeline_state);
void mglRenderBindingSetDepthStencilState(void *binding_state,
                                             void *depth_stencil_state);
int mglRenderBindingGetPipelineState(void *binding_state,
                                        void **pipeline_state_out);
int mglRenderBindingGetDepthStencilState(
    void *binding_state, void **depth_stencil_state_out);
void mglRenderBindingSetCullMode(void *binding_state, uint32_t mode);
void mglRenderBindingSetWinding(void *binding_state, uint32_t winding);
void mglRenderBindingSetDepthBias(void *binding_state,
                                     float bias,
                                     float clamp,
                                     float slope_scale);
void mglRenderBindingSetBlendColor(void *binding_state,
                                      float red,
                                      float green,
                                      float blue,
                                      float alpha);
int mglRenderBindingSetPipelineIfNeeded(void *binding_state,
                                           void *render_encoder,
                                           void *pipeline_state);
int mglRenderBindingSetDepthStencilIfNeeded(void *binding_state,
                                               void *render_encoder,
                                               void *depth_stencil_state);
int mglRenderBindingSetCullIfNeeded(void *binding_state,
                                       void *render_encoder,
                                       uint32_t mode);
int mglRenderBindingSetWindingIfNeeded(void *binding_state,
                                          void *render_encoder,
                                          uint32_t winding);
int mglRenderBindingSetDepthBiasIfNeeded(void *binding_state,
                                            void *render_encoder,
                                            float bias,
                                            float clamp,
                                            float slope_scale);
int mglRenderBindingSetBlendColorIfNeeded(void *binding_state,
                                             void *render_encoder,
                                             float red,
                                             float green,
                                             float blue,
                                             float alpha);
int mglRenderBindingSetPipelineIfNeededForOwner(
    void *binding_state, void *render_encoder_owner, void *pipeline_state);
int mglRenderBindingSetDepthStencilIfNeededForOwner(
    void *binding_state, void *render_encoder_owner,
    void *depth_stencil_state);
int mglRenderBindingSetCullIfNeededForOwner(
    void *binding_state, void *render_encoder_owner, uint32_t mode);
int mglRenderBindingSetWindingIfNeededForOwner(
    void *binding_state, void *render_encoder_owner, uint32_t winding);
int mglRenderBindingSetBlendColorIfNeededForOwner(
    void *binding_state, void *render_encoder_owner,
    float red, float green, float blue, float alpha);
int mglRenderBindingSetTexture(void *binding_state,
                                 void *render_encoder,
                                 void *texture,
                                 uint32_t stage,
                                 uint32_t index);
int mglRenderBindingSetSampler(void *binding_state,
                                 void *render_encoder,
                                 void *sampler,
                                 uint32_t stage,
                                 uint32_t index);
int mglRenderBindingSetTextureForOwner(void *binding_state,
                                         void *render_encoder_owner,
                                         void *texture,
                                         uint32_t stage,
                                         uint32_t index);
int mglRenderBindingSetSamplerForOwner(void *binding_state,
                                         void *render_encoder_owner,
                                         void *sampler,
                                         uint32_t stage,
                                         uint32_t index);
int mglRenderBindingSetDepthBiasIfNeededForOwner(
    void *binding_state,
    void *render_encoder_owner,
    float depth_bias,
    float clamp,
    float slope_scale);
int mglRenderBindingGetTexture(void *binding_state,
                                  uint32_t stage,
                                  uint32_t index,
                                  void **texture_out);
int mglRenderBindingGetSampler(void *binding_state,
                                  uint32_t stage,
                                  uint32_t index,
                                  void **sampler_out);
int mglRenderBindingSetViewport(void *binding_state,
                                  void *render_encoder,
                                  double origin_x,
                                  double origin_y,
                                  double width,
                                  double height,
                                  double znear,
                                  double zfar);
/* Array viewport binding (gl_ViewportIndex): viewports carries count
 * interleaved {x, y, w, h, znear, zfar} tuples, count <= 16. */
int mglRenderBindingSetViewports(void *binding_state,
                                    void *render_encoder,
                                    const double *viewports,
                                    uint64_t count);
int mglRenderBindingSetViewportsForOwner(void *binding_state,
                                            void *render_encoder_owner,
                                            const double *viewports,
                                            uint64_t count);
int mglRenderBindingSetScissor(void *binding_state,
                                 void *render_encoder,
                                 uint64_t x,
                                 uint64_t y,
                                 uint64_t width,
                                 uint64_t height);
int mglRenderBindingSetTriangleFill(void *binding_state,
                                      void *render_encoder,
                                      uint32_t mode);
int mglRenderBindingSetViewportForOwner(void *binding_state,
                                          void *render_encoder_owner,
                                          double origin_x,
                                          double origin_y,
                                          double width,
                                          double height,
                                          double znear,
                                          double zfar);
int mglRenderBindingSetScissorForOwner(void *binding_state,
                                         void *render_encoder_owner,
                                         uint64_t x,
                                         uint64_t y,
                                         uint64_t width,
                                         uint64_t height);
int mglRenderBindingSetTriangleFillForOwner(void *binding_state,
                                              void *render_encoder_owner,
                                              uint32_t mode);
int mglRenderBindingGetStats(void *binding_state,
                               MGLRenderBindingStats *stats_out);

/* Compute encoder setter facade.  These entry points intentionally do not
 * retain resources: the command encoder owns the encoded references, matching
 * Objective-C Metal semantics.  Return 0 on success and -1 for bad inputs. */
int mglRenderSetComputePipelineState(void *compute_encoder,
                                        void *pipeline_state);
int mglRenderSetComputeBuffer(void *compute_encoder,
                                 void *buffer,
                                 uint64_t offset,
                                 uint32_t index);
int mglRenderSetComputeTexture(void *compute_encoder,
                                  void *texture,
                                  uint32_t index);
int mglRenderSetComputeSampler(void *compute_encoder,
                                  void *sampler,
                                  uint32_t index);
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

int mglRenderEncodeComputeBindingSnapshot(
    void *compute_encoder,
    const MGLRenderComputeBindingSnapshot *snapshot,
    char *err,
    size_t errcap);
int mglRenderDispatchCompute(void *compute_encoder,
                                uint32_t groups_x,
                                uint32_t groups_y,
                                uint32_t groups_z,
                                uint32_t threads_x,
                                uint32_t threads_y,
                                uint32_t threads_z);
int mglRenderDispatchComputeIndirect(void *compute_encoder,
                                        void *indirect_buffer,
                                        uint64_t indirect_offset,
                                        uint32_t threads_x,
                                        uint32_t threads_y,
                                        uint32_t threads_z);

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

int mglRenderDispatchComputePlan(
    void *compute_encoder,
    const MGLRenderComputePlan *plan,
    char *err,
    size_t errcap);

/*  compute execution plan: ObjC collects the ordered binding operations
 * and keeps temporary Metal objects alive until this call returns. C++ owns
 * encoder creation, pipeline/binding replay, dispatch, and endEncoding. */
#define MGL_RENDER_COMPUTE_EXECUTION_MAX_OPS 512u
#define MGL_RENDER_COMPUTE_EXECUTION_MAX_DISPATCHES 128u

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
    uint32_t barrier_scope;
} MGLRenderComputeExecutionPlan;

/* Value-state barrier request. These values intentionally mirror Metal's
 * BarrierScope bit values without exposing MTL::* through the C ABI. */
enum {
    MGL_RENDER_COMPUTE_BARRIER_NONE = 0u,
    MGL_RENDER_COMPUTE_BARRIER_BUFFERS = 1u,
    MGL_RENDER_COMPUTE_BARRIER_TEXTURES = 2u,
    MGL_RENDER_COMPUTE_BARRIER_RENDER_TARGETS = 4u,
};

int mglRenderAppendComputeBindingSnapshotToPlan(
    MGLRenderComputeExecutionPlan *plan,
    const MGLRenderComputeBindingSnapshot *snapshot,
    char *err,
    size_t errcap);
int mglRenderAppendComputeDispatchToPlan(
    MGLRenderComputeExecutionPlan *plan,
    const MGLRenderComputePlan *dispatch,
    char *err,
    size_t errcap);

int mglRenderEncodeComputeExecutionPlanForCommandBufferOwner(
    void *command_buffer_owner,
    const MGLRenderComputeExecutionPlan *plan,
    char *err,
    size_t errcap);

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

/* Creates a compute encoder, sets its pipeline and setup bindings, and returns
 * a borrowed encoder owned by the command buffer. Returns -1 on failure. */
int mglRenderBeginComputeDispatch(
    void *command_buffer,
    const MGLRenderComputeDispatchSetup *setup,
    void **compute_encoder_out,
    char *err,
    size_t errcap);
/* Owner-aware form. CommandBufferOwner.current remains inside C++. */
int mglRenderBeginComputeDispatchForCommandBufferOwner(
    void *command_buffer_owner,
    const MGLRenderComputeDispatchSetup *setup,
    void **compute_encoder_out,
    char *err,
    size_t errcap);

/* Dispatches and ends the encoder returned by mglRenderBeginComputeDispatch. */
int mglRenderEndComputeDispatch(void *compute_encoder,
                                   const uint32_t groups[3],
                                   const uint32_t threads[3],
                                   char *err,
                                   size_t errcap);
int mglRenderDispatchComputeThreads(void *compute_encoder,
                                       uint32_t threads_x,
                                       uint32_t threads_y,
                                       uint32_t threads_z,
                                       uint32_t group_x,
                                       uint32_t group_y,
                                       uint32_t group_z);
int mglRenderCreateComputeEncoder(void *command_buffer,
                                     void **compute_encoder_out);
int mglRenderEndComputeEncoder(void *compute_encoder);

/* Command-buffer/render-pass lifecycle facade.  Returned Metal objects are
 * borrowed Objective-C-compatible pointers; the caller retains them through
 * its normal strong state field. */
int mglRenderCreateCommandBuffer(void *command_queue,
                                    void **command_buffer_out);
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

/* Validate and encode a complete compute plan, then (when copy-backs or CPU
 * visibility require a boundary) encode the copy-back blit and perform the
 * owner submit/wait transaction before synchronizing GL CPU prefixes. */
int mglRenderExecuteComputeExecutionPlan(
    void *command_buffer_owner,
    void *recovery_owner,
    const MGLRenderComputeExecutionPlan *plan,
    const MGLRenderCopyBackEntry *copy_backs,
    uint32_t copy_back_count,
    uint32_t require_cpu_visibility,
    MGLRenderComputeExecutionResult *result,
    char *err,
    size_t errcap);

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

/* Snapshot status/error data into caller-owned storage. The completion
 * registration keeps context alive until Metal completes the command buffer,
 * invokes callback once, then invokes destroy_context exactly once. The state
 * pointer passed to callback is valid only for the duration of that call. */
int mglRenderGetCommandBufferState(
    void *command_buffer,
    MGLRenderCommandBufferState *state_out);
const char *mglRenderCommandBufferErrorDescription(
    const MGLRenderCommandBufferState *state);
uint32_t mglRenderCommandBufferStatus(void *command_buffer);
int mglRenderGetCommandBufferLabel(const void *command_buffer,
                                      char *label_out,
                                      size_t label_capacity);
int mglRenderSetCommandBufferLabel(void *command_buffer,
                                      const char *label);
/* Pure value-state classification used by the owner transaction and platform
 * log adapters. Commit classification preserves the legacy status ordering. */
int mglRenderClassifyCommandBufferCommit(
    const MGLRenderCommandBufferState *state,
    MGLRenderCommandBufferCommitDecision *decision_out);
/* Commit one detached/current command buffer through the C++ owner.  When
 * submission_handle points at a matching C++ submission, that ownership is
 * consumed; otherwise the borrowed command buffer is committed directly.
 * Recovery counting, driver-rejection classification, and reset decisions are
 * returned as value-state; the caller only publishes platform logging/reset. */
int mglRenderCommitCommandBufferTransaction(
    void *owner,
    void **submission_handle,
    void *command_buffer,
    void *recovery_owner,
    uint32_t wait_for_completion,
    MGLRenderCommandBufferTransaction *result_out);
int mglRenderClassifyCommandBufferCompletion(
    const MGLRenderCommandBufferState *state,
    MGLRenderCommandBufferCompletionDecision *decision_out);
/* Thread-safe owner for the renderer's command-completion error counters.
 * Timestamps are caller-provided seconds so policy remains independent of
 * Foundation and can be tested deterministically. */
int mglRenderCreateCommandRecoveryOwner(void **owner_out);
void mglRenderDestroyCommandRecoveryOwner(void **owner);
int mglRenderCommandRecoveryRecordError(
    void *owner,
    double now,
    MGLRenderCommandRecoverySnapshot *state_out);
/* Platform exception boundary for failures that cannot cross the C++ ABI.
 * Applies the recovery update at most once to transaction_inout. */
int mglRenderCommandRecoveryRecordTransactionFailure(
    void *owner,
    const MGLRenderCommandBufferState *state,
    MGLRenderCommandBufferTransaction *transaction_inout);
int mglRenderCommandRecoveryRecordSuccess(
    void *owner,
    double now,
    MGLRenderCommandRecoverySuccess *result_out);
/* Kept separate from RecordSuccess to preserve the legacy two-lock completion
 * sequence. Returns 1 when recovery mode was cleared, 0 when already clear. */
int mglRenderCommandRecoveryClearMode(void *owner);
int mglRenderCommandRecoveryShouldSkip(
    void *owner,
    double now,
    MGLRenderCommandRecoverySkipDecision *decision_out);
/* Classify one completed command buffer and apply the legacy recovery-owner
 * update sequence. Success intentionally performs RecordSuccess followed by
 * the separate ClearMode operation so the former two-lock ordering remains
 * observable; ObjC consumes the returned value state for logging/reset work. */
int mglRenderProcessCommandBufferCompletion(
    void *owner,
    const MGLRenderCommandBufferState *state,
    double now,
    MGLRenderCommandBufferCompletionResult *result_out);
/* Register the standard command-recovery completion handler without capturing
 * an Objective-C renderer.  The C++ recovery owner records error/success
 * counters and latches a deferred-reset request for the GL thread. */
int mglRenderAddCommandBufferRecoveryCompletion(
    void *command_buffer,
    void *recovery_owner);
/* Consume a reset request latched by a completion worker. Returns 1 when a
 * request was consumed, 0 when none is pending, and -1 for invalid owner. */
int mglRenderCommandRecoveryTakeResetRequest(void *recovery_owner);
int mglRenderAddCommandBufferCompletion(
    void *command_buffer,
    MGLRenderCommandBufferCompletion callback,
    void *context,
    MGLRenderDestroyContext destroy_context);
/* Register a completion on CommandBufferOwner.current without exposing the
 * borrowed command buffer through the C ABI. */
int mglRenderAddCommandBufferOwnerCompletion(
    void *owner,
    MGLRenderCommandBufferCompletion callback,
    void *context,
    MGLRenderDestroyContext destroy_context);
/* The current-buffer owner retains the autoreleased command buffer returned
 * by Metal. Detach moves that +1 reference into a submission handle; commit
 * consumes the submission only after Metal accepts it. Returned command
 * buffer pointers are borrowed. */
int mglRenderCreateCommandBufferOwner(void *command_queue,
                                         void **owner_out,
                                         void **command_buffer_out);
/* Adopt an existing (ObjC-created) command buffer as the owner's current —
 * gate-off fallback so the owner stays the single source on both gates.
 * Returns 0 with *owner_out set (the owner retains the buffer). */
int mglRenderCreateCommandBufferOwnerAdopt(void *command_buffer,
                                              void **owner_out);
/* Borrowed pointer to the owner's current command buffer (NULL when the
 * owner has none / owner is NULL). */
void *mglRenderCommandBufferOwnerGetCurrent(void *owner);
/* Returns 1 when current exists, 0 when empty, and -1 for a null owner. */
int mglRenderCommandBufferOwnerHasCurrent(void *owner);
/* Create the next current command buffer from the queue retained by the
 * owner.  Returns 0 on success, 1 when the owner has no queue (adopted
 * fallback), and -1 on allocation/argument failure. */
int mglRenderCommandBufferOwnerCreateNext(void *owner,
                                             void **command_buffer_out);
/* Snapshot the owner's current buffer without exposing it to the caller.
 * Returns -1 when the owner/current buffer/state output is missing. */
int mglRenderGetCommandBufferOwnerState(
    void *owner,
    MGLRenderCommandBufferState *state_out);
/* Boolean convenience form: returns 1 when a snapshot was produced. */
int mglRenderCommandBufferOwnerHasState(
    void *owner,
    MGLRenderCommandBufferState *state_out);
/* The owner retains the most recently accepted submission. These APIs keep
 * glFinish/readback synchronization in the lifecycle owner without exposing
 * a borrowed command-buffer pointer to Objective-C. */
int mglRenderCommandBufferOwnerHasLastSubmitted(void *owner);
/* Wait for one submitted command buffer and return a value-state snapshot.
 * Returns 0 on completed success, 1 when the buffer is still NotEnqueued,
 * and -1 for invalid arguments, wait failures, or command-buffer errors. */
int mglRenderWaitCommandBufferState(
    void *command_buffer,
    MGLRenderCommandBufferState *state_out);
int mglRenderWaitCommandBufferOwnerLastSubmitted(
    void *owner,
    MGLRenderCommandBufferState *state_out);
/* Encode presentation on the owner's current not-enqueued command buffer.
 * Returns 0 on success, 1 when the current buffer is already finalized, and
 * -1 for missing owner/current buffer/drawable. */
int mglRenderPresentDrawableForCommandBufferOwner(
    void *owner,
    void *drawable,
    MGLRenderCommandBufferState *state_out);
int mglRenderEncodeWaitForEventForCommandBufferOwner(
    void *owner, void *event, uint64_t value);
int mglRenderResetCommandBufferOwner(void *owner,
                                        void *command_queue,
                                        void **command_buffer_out);
void mglRenderDiscardCommandBufferOwnerCurrent(void *owner);
/* Reentrancy guard for command-buffer commit. Returns 1 when acquired, 0
 * when a commit is already in progress, and -1 for a missing owner. This
 * preserves the former MGLCommandState BOOL semantics; it is intentionally
 * not a cross-thread synchronization primitive. */
int mglRenderCommandBufferOwnerBeginCommit(void *owner);
void mglRenderCommandBufferOwnerEndCommit(void *owner);
/* Consume the marker for a current buffer created by the preceding submit
 * transaction. Returns 1 and a borrowed current buffer once, 0 when no such
 * buffer is pending, and -1 for invalid arguments. */
int mglRenderCommandBufferOwnerConsumeTransactionCurrent(
    void *owner,
    void **command_buffer_out);
int mglRenderTakeCommandBufferSubmission(void *owner,
                                             void **submission_out,
                                             void **command_buffer_out);
int mglRenderCommitCommandBufferSubmission(void **submission);
void mglRenderDestroyCommandBufferSubmission(void **submission);
void mglRenderDestroyCommandBufferOwner(void **owner);
/* The opaque owner holds the +1 Metal-cpp command-queue reference. The queue
 * pointer is borrowed and may be assigned to an ObjC strong field during the
 * migration. max_command_buffers=0 selects Metal's default configuration. */
int mglRenderCreateCommandQueueOwner(uint32_t max_command_buffers,
                                        void **owner_out,
                                        void **command_queue_out);
int mglRenderResetCommandQueueOwner(void *owner,
                                       uint32_t max_command_buffers,
                                       void **command_queue_out);
void mglRenderDestroyCommandQueueOwner(void **owner);
/* Per-command-buffer MDI argument arena. The opaque owner keeps the sole
 * persistent +1 reference; returned buffers are borrowed migration views. */
int mglRenderCreateMDIScratchOwner(void **owner_out);
int mglRenderAllocateMDIScratch(void *owner,
                                   uint64_t length,
                                   uint64_t alignment,
                                   void **buffer_out,
                                   uint64_t *offset_out,
                                   uint64_t *capacity_out);
void mglRenderResetMDIScratchOwner(void *owner);
void mglRenderDestroyMDIScratchOwner(void **owner);
int mglRenderCommitCommandBuffer(void *command_buffer);
int mglRenderWaitCommandBuffer(void *command_buffer);

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

int mglRenderEncodeBindingSnapshot(
    void *render_encoder,
    const MGLRenderBindingSnapshot *snapshot,
    char *err,
    size_t errcap);
int mglRenderEncodeBindingSnapshotForRenderEncoderOwner(
    void *render_encoder_owner,
    const MGLRenderBindingSnapshot *snapshot,
    char *err,
    size_t errcap);

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

int mglRenderEncodeResourceBindingSnapshot(
    void *binding_state,
    void *render_encoder,
    const MGLRenderResourceBindingSnapshot *snapshot,
    char *err,
    size_t errcap);
int mglRenderEncodeResourceBindingSnapshotForRenderEncoderOwner(
    void *binding_state,
    void *render_encoder_owner,
    const MGLRenderResourceBindingSnapshot *snapshot,
    char *err,
    size_t errcap);

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

/* Initialize a value state with Metal's render-pass descriptor defaults. */
void mglRenderInitDefaultRenderPassState(
    MGLRenderPassState *state_out);

typedef enum MGLRenderPassAttachmentKind_t {
    MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR = 0,
    MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH = 1,
    MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL = 2,
} MGLRenderPassAttachmentKind;

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

/* Persistent render-pass identity and FBO cache. The owner is authoritative
 * for Metal-cpp mode; ObjC fields remain a synchronized migration view. */
int mglRenderCreateRenderPassIdentityOwner(void **owner_out);
int mglRenderUpdateRenderPassIdentity(
    void *owner, const MGLRenderPassIdentityState *state);
int mglRenderGetRenderPassIdentity(
    void *owner, MGLRenderPassIdentityState *state_out);
int mglRenderSetFboMatchCache(
    void *owner, const MGLRenderFboMatchCacheState *cache);
int mglRenderGetFboMatchCache(
    void *owner, MGLRenderFboMatchCacheState *cache_out);
void mglRenderClearFboMatchCache(void *owner);
void mglRenderDestroyRenderPassIdentityOwner(void **owner);

/* Persistent value-state owner for render-pass attachment/dimension fields.
 * The owner retains every attachment/resolve/visibility/rate-map resource
 * referenced by the snapshot and releases replaced resources on update. */
int mglRenderCreateRenderPassStateOwner(
    const MGLRenderPassState *state, void **owner_out);
int mglRenderCreateDefaultRenderPassStateOwner(void **owner_out);
int mglRenderSetRenderPassStateAttachment(
    void *owner,
    uint32_t attachment_kind,
    uint32_t color_index,
    const MGLRenderPassAttachmentState *attachment);
int mglRenderSetRenderPassStateAttachmentTexture(
    void *owner,
    uint32_t attachment_kind,
    uint32_t color_index,
    void *texture,
    uint64_t level,
    uint64_t slice,
    uint64_t depth_plane,
    uint32_t layered);
int mglRenderSetRenderPassStateAttachmentActions(
    void *owner,
    uint32_t attachment_kind,
    uint32_t color_index,
    uint32_t load_action,
    uint32_t store_action,
    uint64_t store_action_options);
int mglRenderSetRenderPassStateColorClear(
    void *owner,
    uint32_t color_index,
    double red,
    double green,
    double blue,
    double alpha);
int mglRenderSetRenderPassStateDepthClear(
    void *owner, double clear_depth);
int mglRenderSetRenderPassStateStencilClear(
    void *owner, uint32_t clear_stencil);
int mglRenderSetRenderPassStateVisibility(
    void *owner, void *visibility_result_buffer,
    uint32_t visibility_result_type);
int mglRenderSetRenderPassStateDimensions(
    void *owner, uint64_t width, uint64_t height);
/* pending shared-event slot inside the C++ owner.
 * `int` in these decls is GLsizei (GL signed 32-bit) — the C ABI matches. */
int mglRenderCreatePendingEventOwner(void **owner_out);
int mglRenderPendingEventPrepare(void *owner_handle, int sync_name,
                                    void **event_out);
int mglRenderPendingEventDetach(void *owner_handle,
                                   int *sync_name_out, void **event_out);
void mglRenderPendingEventClear(void *owner_handle);
void mglRenderDestroyPendingEventOwner(void **owner_handle);
/* detached-submission ownership guard. */
int mglRenderCommandBufferSubmissionMatchesBuffer(void *submission_handle,
                                                     void *command_buffer);
/* current-CB sync tracking list inside the C++ owner. */
int mglRenderCommandBufferOwnerAppendSync(void *owner_handle, Sync *sync);
void mglRenderCommandBufferOwnerClearSyncs(void *owner_handle);
int mglRenderGetRenderPassStateOwner(
    void *owner, MGLRenderPassState *state_out);
/* Returns a borrowed attachment snapshot. Object pointers remain owned by the
 * render-pass owner and are valid only while that owner keeps the state. */
int mglRenderGetRenderPassAttachmentStateOwner(
    void *owner,
    uint32_t attachment_kind,
    uint32_t color_index,
    MGLRenderPassAttachmentState *attachment_out);
int mglRenderCreateRenderEncoderFromStateOwner(
    void *command_buffer, void *state_owner, void **render_encoder_out);
/* Owner-aware variant used by command-lifecycle callers. The command buffer
 * stays inside CommandBufferOwner; the returned encoder is borrowed. */
int mglRenderCreateRenderEncoderFromCommandBufferOwnerState(
    void *command_buffer_owner,
    const MGLRenderPassState *render_pass,
    void **render_encoder_out);
/* Borrowed-object convenience forms for Objective-C callers.  The return
 * value is an opaque Metal object owned by the command-buffer owner. */
void *mglRenderCreateRenderEncoderBorrowed(
    void *command_buffer_owner,
    const MGLRenderPassState *render_pass);
void *mglRenderCreateBlitEncoderBorrowed(void *command_buffer_owner);
void *mglRenderCreateComputeEncoderBorrowed(void *command_buffer_owner);
void mglRenderDestroyRenderPassStateOwner(void **owner);

/* C++ owns the temporary MTL::RenderPassDescriptor used to create the
 * borrowed render encoder. Attachment resources remain caller-owned. */
int mglRenderCreateRenderEncoderFromState(
    void *command_buffer,
    const MGLRenderPassState *render_pass,
    void **render_encoder_out);
void *mglRenderGetRenderPassAttachmentTextureOwner(
    void *owner, uint32_t attachment_kind, uint32_t color_index);
int mglRenderGetRenderPassAttachmentSubresourceOwner(
    void *owner, uint32_t attachment_kind, uint32_t color_index,
    uint64_t *level_out, uint64_t *slice_out, uint64_t *depth_plane_out);
int mglRenderGetRenderTargetSizeOwner(
    void *owner, uint64_t *width_out, uint64_t *height_out);
int mglRenderPassUsesColorTextureOwner(
    void *owner, void *texture, uint32_t *attachment_index_out);
int mglRenderGetRenderPassAttachmentActionsOwner(
    void *owner, uint32_t attachment_kind, uint32_t color_index,
    uint32_t *load_action_out, uint32_t *store_action_out,
    uint64_t *store_action_options_out);
uint32_t mglRenderPassLoadActionForTrace(
    void *owner, uint32_t attachment_kind, uint32_t color_index,
    uint32_t default_load_action);
uint32_t mglRenderPassStoreActionForTrace(
    void *owner, uint32_t attachment_kind, uint32_t color_index,
    uint32_t default_store_action);
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
int mglRenderEncodeColorClearForCommandBufferOwner(
    void *command_buffer_owner,
    void *texture,
    uint64_t level,
    uint64_t slice,
    uint64_t depth_plane,
    double red,
    double green,
    double blue,
    double alpha);
int mglRenderEncodeDepthClear(void *command_buffer,
                                 void *texture,
                                 uint64_t level,
                                 uint64_t slice,
                                 uint64_t depth_plane,
                                 double clear_depth);
int mglRenderEncodeDepthClearForCommandBufferOwner(
    void *command_buffer_owner,
    void *texture,
    uint64_t level,
    uint64_t slice,
    uint64_t depth_plane,
    double clear_depth);
int mglRenderEncodeMultisampleResolve(
    void *command_buffer,
    uint32_t attachment_kind,
    void *source_texture,
    uint64_t source_level,
    uint64_t source_slice,
    uint64_t source_depth_plane,
    void *resolve_texture,
    uint64_t resolve_level,
    uint64_t resolve_slice,
    uint64_t resolve_depth_plane,
    uint32_t resolve_filter);
int mglRenderEncodeMultisampleResolveForCommandBufferOwner(
    void *command_buffer_owner,
    uint32_t attachment_kind,
    void *source_texture,
    uint64_t source_level,
    uint64_t source_slice,
    uint64_t source_depth_plane,
    void *resolve_texture,
    uint64_t resolve_level,
    uint64_t resolve_slice,
    uint64_t resolve_depth_plane,
    uint32_t resolve_filter);
/* The owner retains the autoreleased encoder returned by Metal. End is
 * idempotent per owned encoder; destroy releases the retained reference. */
int mglRenderCreateRenderEncoderOwnerFromState(
    void *command_buffer,
    const MGLRenderPassState *render_pass,
    void **owner_out,
    void **render_encoder_out);
int mglRenderResetRenderEncoderOwnerFromState(
    void *owner,
    void *command_buffer,
    const MGLRenderPassState *render_pass,
    void **render_encoder_out);
int mglRenderCreateRenderEncoderOwner(
    void *render_encoder,
    void **owner_out);
int mglRenderResetRenderEncoderOwner(
    void *owner,
    void *render_encoder);
int mglRenderEndRenderEncoderOwner(void *owner);
int mglRenderSetRenderEncoderOwnerLabel(void *owner,
                                           const char *label);
int mglRenderEncoderOwnerHasCurrent(void *owner);
void mglRenderDestroyRenderEncoderOwner(void **owner);
int mglRenderEndRenderEncoder(void *render_encoder);
int mglRenderCreateBlitEncoder(void *command_buffer,
                                  void **blit_encoder_out);
/* Creates a borrowed blit encoder from CommandBufferOwner.current without
 * exposing the current command buffer through the C ABI. */
int mglRenderCreateBlitEncoderFromCommandBufferOwner(
    void *command_buffer_owner,
    void **blit_encoder_out);
/* Encode a complete texture-to-texture preservation copy inside the owner.
 * Every common array slice and mip level is copied at its full mip extent;
 * encoder creation and endEncoding remain entirely in C++. */
int mglRenderCopyMatchingTextureSubresourcesForCommandBufferOwner(
    void *command_buffer_owner,
    void *source_texture,
    void *destination_texture);
typedef struct MGLRenderBufferCopyEntry_t {
    void *source_buffer;
    uint64_t source_offset;
    void *destination_buffer;
    uint64_t destination_offset;
    uint64_t length;
} MGLRenderBufferCopyEntry;
int mglRenderEncodeBufferCopiesForCommandBufferOwner(
    void *command_buffer_owner,
    const MGLRenderBufferCopyEntry *entries,
    uint32_t entry_count);
int mglRenderCreateComputeEncoderFromCommandBufferOwner(
    void *command_buffer_owner,
    void **compute_encoder_out);
int mglRenderEndBlitEncoder(void *blit_encoder);
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
int mglRenderEncodeTextureUploadLayersForCommandBufferOwner(
    void *command_buffer_owner,
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
int mglRenderBlitCopyBuffer(void *blit_encoder,
                               void *source_buffer,
                               uint64_t source_offset,
                               void *destination_buffer,
                               uint64_t destination_offset,
                               uint64_t size);
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
int mglRenderBlitSynchronizeTexture(void *blit_encoder,
                                       void *texture,
                                       uint64_t slice,
                                       uint64_t level);
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
int mglRenderBlitGenerateMipmaps(void *blit_encoder,
                                    void *texture);

/* Render draw command facade. Enum values are passed as uint32_t so the C ABI
 * remains independent of Metal headers. Resources are borrowed for encoding. */
int mglRenderDrawPrimitives(void *render_encoder,
                               uint32_t primitive_type,
                               uint64_t vertex_start,
                               uint64_t vertex_count,
                               uint64_t instance_count,
                               uint64_t base_instance);
int mglRenderDrawIndexedPrimitives(void *render_encoder,
                                      uint32_t primitive_type,
                                      uint64_t index_count,
                                      uint32_t index_type,
                                      void *index_buffer,
                                      uint64_t index_buffer_offset,
                                      uint64_t instance_count,
                                      int64_t base_vertex,
                                      uint64_t base_instance);
int mglRenderDrawPrimitivesIndirect(void *render_encoder,
                                       uint32_t primitive_type,
                                       void *indirect_buffer,
                                       uint64_t indirect_buffer_offset);
int mglRenderDrawIndexedPrimitivesIndirect(
    void *render_encoder,
    uint32_t primitive_type,
    uint32_t index_type,
    void *index_buffer,
    uint64_t index_buffer_offset,
    void *indirect_buffer,
    uint64_t indirect_buffer_offset);

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

/* Encodes one draw. render_encoder is borrowed. Invalid plans return -1 and
 * populate err without encoding a partial draw. */
int mglRenderEncodeDraw(void *render_encoder,
                           const MGLRenderDrawPlan *plan,
                           char *err,
                           size_t errcap);
int mglRenderEncodeDrawForRenderEncoderOwner(
    void *render_encoder_owner,
    const MGLRenderDrawPlan *plan,
    char *err,
    size_t errcap);

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
int mglRenderCreateCullDistanceIndexPlan(
    void *device,
    const void *source_indices,
    uint32_t source_index_type,
    uint64_t source_index_count,
    uint32_t draw_mode,
    int primitive_restart_enabled,
    uint32_t primitive_restart_index,
    int64_t base_vertex,
    int polygon_line_mode,
    void **owner_out,
    void **index_buffer_out,
    uint64_t *primitive_count_out);
int mglRenderGetCullDistanceIndexPrimitive(
    void *owner,
    uint64_t primitive_index,
    MGLRenderCullDistancePrimitive *primitive_out);
void mglRenderDestroyCullDistanceIndexPlan(void **owner);

int mglRenderSetRenderBuffer(void *render_encoder,
                                void *buffer,
                                uint64_t offset,
                                uint32_t stage,
                                uint32_t index);
int mglRenderSetRenderBytes(void *render_encoder,
                               const void *bytes,
                               size_t length,
                               uint32_t stage,
                               uint32_t index);
int mglRenderSetRenderPipelineState(void *render_encoder,
                                       void *pipeline_state);
int mglRenderSetRenderDepthStencilState(void *render_encoder,
                                           void *depth_stencil_state);
int mglRenderSetRenderTexture(void *render_encoder,
                                 void *texture,
                                 uint32_t stage,
                                 uint32_t index);
int mglRenderSetRenderSampler(void *render_encoder,
                                 void *sampler,
                                 uint32_t stage,
                                 uint32_t index);
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
int mglRenderSetTessellationFactorBuffer(void *render_encoder,
                                            void *buffer,
                                            uint64_t offset,
                                            uint64_t instance_stride);
int mglRenderSetRenderBufferForOwner(void *render_encoder_owner,
                                        void *buffer,
                                        uint64_t offset,
                                        uint32_t stage,
                                        uint32_t index);
int mglRenderSetRenderBytesForOwner(void *render_encoder_owner,
                                       const void *bytes,
                                       size_t length,
                                       uint32_t stage,
                                       uint32_t index);
int mglRenderSetRenderPipelineStateForOwner(void *render_encoder_owner,
                                               void *pipeline_state);
int mglRenderSetRenderDepthStencilStateForOwner(void *render_encoder_owner,
                                                   void *depth_stencil_state);
int mglRenderSetRenderTextureForOwner(void *render_encoder_owner,
                                         void *texture,
                                         uint32_t stage,
                                         uint32_t index);
int mglRenderSetRenderSamplerForOwner(void *render_encoder_owner,
                                         void *sampler,
                                         uint32_t stage,
                                         uint32_t index);
int mglRenderSetRenderViewportForOwner(void *render_encoder_owner,
                                          double origin_x,
                                          double origin_y,
                                          double width,
                                          double height,
                                          double znear,
                                          double zfar);
int mglRenderSetRenderScissorForOwner(void *render_encoder_owner,
                                         uint64_t x,
                                         uint64_t y,
                                         uint64_t width,
                                         uint64_t height);
int mglRenderSetDepthClipModeForOwner(void *render_encoder_owner,
                                         uint32_t mode);
int mglRenderSetStencilReferenceValuesForOwner(
    void *render_encoder_owner,
    uint32_t front_reference,
    uint32_t back_reference);
int mglRenderSetTessellationFactorBufferForOwner(
    void *render_encoder_owner,
    void *buffer,
    uint64_t offset,
    uint64_t instance_stride);
int mglRenderDrawPatches(void *render_encoder,
                            uint64_t control_point_count,
                            uint64_t patch_start,
                            uint64_t patch_count,
                            void *patch_index_buffer,
                            uint64_t patch_index_buffer_offset,
                            uint64_t instance_count,
                            uint64_t base_instance);
int mglRenderDrawIndexedPatches(void *render_encoder,
                                   uint64_t control_point_count,
                                   uint64_t patch_start,
                                   uint64_t patch_count,
                                   void *patch_index_buffer,
                                   uint64_t patch_index_buffer_offset,
                                   void *control_point_index_buffer,
                                   uint64_t control_point_index_buffer_offset,
                                   uint64_t instance_count,
                                   uint64_t base_instance);

int mglRenderCreateIndirectCommandBuffer(
    uint32_t command_types,
    int inherit_pipeline_state,
    int inherit_buffers,
    uint32_t max_vertex_buffer_bind_count,
    uint32_t max_fragment_buffer_bind_count,
    uint64_t max_command_count,
    uint64_t resource_options,
    void **indirect_buffer_out);
int mglRenderResetIndirectCommandBuffer(void *indirect_buffer,
                                           uint64_t location,
                                           uint64_t length);
int mglRenderGetIndirectRenderCommand(void *indirect_buffer,
                                         uint64_t command_index,
                                         void **command_out);
int mglRenderSetIndirectDrawIndexed(void *indirect_command,
                                       uint32_t primitive_type,
                                       uint64_t index_count,
                                       uint32_t index_type,
                                       void *index_buffer,
                                       uint64_t index_buffer_offset,
                                       uint64_t instance_count,
                                       int64_t base_vertex,
                                       uint64_t base_instance);
int mglRenderSetIndirectDraw(void *indirect_command,
                                uint32_t primitive_type,
                                uint64_t vertex_start,
                                uint64_t vertex_count,
                                uint64_t instance_count,
                                uint64_t base_instance);
int mglRenderUseRenderResource(void *render_encoder,
                                  void *resource,
                                  uint32_t usage,
                                  uint32_t stages);
int mglRenderExecuteIndirectCommands(void *render_encoder,
                                        void *indirect_buffer,
                                        uint64_t location,
                                        uint64_t length);
int mglRenderReplayBatchDrawsForRenderEncoderOwner(
    void *render_encoder_owner,
    const MGLRenderReplayBatch *batch,
    char *err,
    size_t errcap);
int mglRenderUseRenderResourceForOwner(void *render_encoder_owner,
                                          void *resource,
                                          uint32_t usage,
                                          uint32_t stages);
int mglRenderExecuteIndirectCommandsForOwner(void *render_encoder_owner,
                                                 void *indirect_buffer,
                                                 uint64_t location,
                                                 uint64_t length);

#ifdef __cplusplus
} // extern "C"
#endif
