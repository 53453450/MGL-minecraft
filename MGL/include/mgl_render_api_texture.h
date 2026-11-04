/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef MGL_RENDER_API_TEXTURE_H
#define MGL_RENDER_API_TEXTURE_H

/* Declarations for the texture slice of the renderer facade.
 * Value layouts are defined here; types of other subsystems that are only
 * passed by pointer are incomplete (mgl_render_fwd.h). */

#include "mgl_render_fwd.h"

#ifdef __cplusplus
extern "C" {
#endif

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

uint32_t mglRenderRepairedDefaultStencilFormat(uint32_t stencil_format);

int mglRenderTargetIsRenderbuffer(uint32_t target);

int mglRenderMSSamplePlaneAdjust(int in_ms_loop, uint32_t target,
                                 int32_t offset);

int mglRenderFilterIsNearest(uint32_t filter);

uint32_t mglRenderNearestFilter(void);

int mglRenderCubeMapFaceSlice(uint32_t textarget, uint32_t *out);

int mglRenderAttachmentUsesArrayLayer(uint32_t textarget);

int mglRenderPackedDepthStencilFormat(uint32_t internalformat);

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

/* stored color-component count for an internal format.  Unknown formats → 4. */
uint32_t mglRenderStoredColorComponents(uint32_t internal_format);

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
 * range cannot leave a partially encoded layer prefix.  flip_y writes source
 * row r to Metal row mip_height-1-(destination_y+r). */
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
    uint64_t destination_z,
    int flip_y);

int mglRenderEncodeTextureUploadLayersForCommandBufferOwner(MGLCommandBufferOwner *command_buffer_owner, void *source_buffer, uint64_t source_offset, uint64_t source_bytes_per_row, uint64_t source_bytes_per_image, uint64_t source_layer_stride, uint64_t source_width, uint64_t source_height, uint64_t source_depth, void *destination_texture, uint64_t destination_base_slice, uint64_t layer_count, uint64_t destination_level, uint64_t destination_x, uint64_t destination_y, uint64_t destination_z, int flip_y);

int mglRenderEncodePackedDepthStencilUploadForCommandBufferOwner(MGLCommandBufferOwner *command_buffer_owner, void *source_buffer, uint64_t width, uint64_t height, void *destination_texture, uint64_t destination_slice, uint64_t destination_level, uint64_t destination_x, uint64_t destination_y, int flip_y);


/* C++-owned transient upload buffer. The returned buffer is borrowed from
 * the opaque owner; Metal command encoders retain it when a copy command is
 * recorded, so the owner may be destroyed immediately after encoding. */
int mglRenderCreateTextureStagingOwner(const void *bytes, uint64_t length, uint64_t resource_options, MGLTextureStagingOwner **owner_out, void **buffer_out);

void mglRenderDestroyTextureStagingOwner(MGLTextureStagingOwner **owner);
/* Return the staging owner to the pool only once `command_buffer` completes:
 * the upload blit keeps reading those bytes until the GPU is finished with it,
 * and the upload command buffer is committed without being waited on. */
int mglRenderDestroyTextureStagingOwnerAfterCommandBuffer(
    MGLTextureStagingOwner **owner, void *command_buffer);

int mglRenderGetTextureInfo(const void *texture,
                               MGLRenderTextureInfo *info_out);

int mglRenderTextureIsFramebufferOnly(const void *texture);

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

/* Sparse (ARB_sparse_texture) textures.  Creation hands back both the
 * placement heap and the placement-sparse texture: the page commitment path
 * needs the heap again when it maps tiles. */
int mglRenderCreateSparseTextureFromState(
    const MGLRenderTextureDescriptorState *texture_descriptor,
    const char *label,
    void **heap_out,
    void **texture_out);

/* Commit (commit != 0) or decommit a texel-space region.  The region must be
 * page aligned; the page is GL_VIRTUAL_PAGE_SIZE_{X,Y,Z} = 64.
 * heap_offset is the starting placement-heap tile index for Map (ignored on
 * Unmap).  Callers must give each Map a non-overlapping range. */
int mglRenderSparseTextureCommit(void *heap, void *texture,
                                 uint32_t x, uint32_t y, uint32_t z,
                                 uint32_t width, uint32_t height, uint32_t depth,
                                 uint32_t level, int commit,
                                 uint64_t heap_offset);

/* GPU-blit zeros into a mapped sparse region (Private textures). */
int mglRenderSparseTextureClearRegion(void *texture,
                                      uint32_t x, uint32_t y, uint32_t z,
                                      uint32_t width, uint32_t height,
                                      uint32_t depth, uint32_t level);

/* One page-aligned Unmap region for batched scrub (one CB for many pages). */
typedef struct MGLSparseUnmapRegion_t {
    uint32_t x, y, z;
    uint32_t width, height, depth;
    uint32_t level;
} MGLSparseUnmapRegion;

int mglRenderSparseTextureUnmapRegions(void *heap, void *texture,
                                       const MGLSparseUnmapRegion *regions,
                                       uint32_t count);

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

/* Apply GL BASE_LEVEL/MAX_LEVEL and swizzle state to a sampled texture. The
 * returned texture is +1 retained for the caller; the Texture cache keeps its
 * own reference. */
int mglRenderSampledTextureViewForBaseLevel(
    Texture *texture_object,
    void *source_texture,
    void **view_out);

/* Pixel format samplers read from a texture of `pixel_format`: its stencil
 * aspect in DEPTH_STENCIL_TEXTURE_MODE == STENCIL_INDEX. */
uint32_t mglRenderSampledViewPixelFormat(uint32_t depth_stencil_mode,
                                         uint32_t pixel_format);

int mglRenderTextureTargetPlan(
    uint32_t gl_target,
    uint32_t sample_count,
    MGLRenderTextureTargetPlan *plan_out);

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

uint32_t mglRenderTextureDataKindForPixelFormat(uint32_t pixel_format);

/* pure pixel-format and GL internal-format predicates.
 * The C ABI carries only stable integer enum values; ObjC compatibility
 * headers remain thin wrappers around these C++ tables. */
/* compressed upload row math.  Returns the block height
 * and rounded upload-row count using uint64_t so the C ABI is Foundation-free. */
/* data-kind → debug name string (static literals).
 * kind uses MGL_RENDER_TEXTURE_DATA_KIND_*. */
const char *mglRenderTextureDataKindName(uint32_t kind);

/* min-filter → uses-mipmaps.  Returns 1/0. */
int mglRenderTextureMinFilterUsesMipmaps(uint32_t min_filter);

/* layer / sRGB pixel-format tables.  Pixel format
 * is the Apple MGLPixelFormat numeric value.  Effective honors
 * GL_EXT_texture_sRGB_decode via the raw srgb_decode_ext enum. */
int mglRenderMetalLayerPixelFormatIsSupported(uint32_t pixel_format);

uint32_t mglRenderLinearPixelFormat(uint32_t pixel_format);

uint32_t mglRenderEffectiveMTLPixelFormat(uint32_t pixel_format,
                                             uint32_t srgb_decode_ext);

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

/* SNORM8 texture bytes -> GL format/type, bypassing
 * the lossy BGRA8 UNORM intermediate.  Mirrors the ObjC sourceIsSnorm8
 * path (1 on success, 0 on bad args / unsupported format). */
int mglRenderCopySnorm8TextureBytesToGL(
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

int mglRenderTextureUploadRoute(uint32_t texture_type,
                                   uint32_t storage_mode,
                                   int has_agx_3d_copy_bug);

/* Metal mipmap level dimension — the greatest
 * 2^(level) divisor of base (base>>level, clamped to 1).  Pure computation
 * shared by both gates (the ObjC mglMetalTextureLevelDimension keeps the
 * extern linkage its many callers use). */
uint64_t mglRenderMetalTextureLevelDimension(uint64_t base, uint64_t level);

/* O3.3: ImageBindPixelFormat..ImageViewSliceCount -> mgl_binding_texture.h */
int mglRenderIsTextureBufferTarget(uint32_t gl_target);

int mglRenderTextureDimsValid(uint32_t gl_target, int32_t width, int32_t height,
                              int32_t depth);

int mglRenderTextureBufferNeedsDirty(int is_tbo, int has_buf, uint32_t buf_dirty);

int mglRenderTextureNameIsDefault(uint32_t name);

int mglRenderMSTextureUnitIndex(int image_arrayed);

int mglRenderIsMultisampleTextureTarget(uint32_t gl_target);

int mglRenderRejectDefaultTypedTexture(int typed_is_default, int active_is_real);

int mglRenderImageDimIsBuffer(uint32_t image_dim);

int mglRenderExpectedTypeIsTextureBuffer(uint32_t expected_type);

int mglRenderPlanTexelBuffer2DSize(uint64_t texel_count, uint32_t max_texture_size,
                                   uint32_t *width_out, uint32_t *height_out);

uint32_t mglRenderFallbackSampledTextureType(uint32_t expected_type);

uint32_t mglRenderFallbackSampledPixelFormat(uint32_t data_kind);

int mglRenderTextureUsageForAccess(uint32_t gl_access, uint32_t *usage_out);

int mglRenderPixelFormatNeedsShaderAtomic(uint32_t pixel_format);

int mglRenderTextureArrayDepthForType(uint32_t tex_type, int is_array,
                                      int ms_emulated, uint64_t width,
                                      uint64_t height, uint64_t depth,
                                      uint64_t *array_out, uint64_t *depth_out);

uint32_t mglRenderTextureDescHeight(uint32_t tex_type, uint32_t height);

/* C1: binding slot/sampler/stage/plain-uniform -> mgl_binding_policy.h */
/* O3.3: stage UBO/SSBO/attrib bind plan + helpers -> mgl_binding_stage.h */
/* O3.3: sampled/storage/Y-flip/sampler-materialize -> mgl_binding_texture.h */
/* C1: format-class PSO topology/blend/stencil/viewport -> mgl_pso_format_class.h */
int mglRenderTextureTargetIsBuffer(uint32_t target);

int mglRenderTextureTargetIsArrayOr3D(uint32_t target);

int mglRenderTextureTargetIs2D(uint32_t target);

int mglRenderTextureTargetIs3D(uint32_t target);

int mglRenderImageAccessIsReadOnly(uint32_t access);

int mglRenderImageUnitSliceNeedsFlush(int has_tex, int has_view, int layered,
                                      uint32_t target, uint32_t access);

uint32_t mglRenderFallbackPixelFormat(uint32_t mapped, uint32_t internalformat);

int mglRenderTextureTargetIsArray(uint32_t target);

uint32_t mglRenderBytesPerPixelForInternalFormat(uint32_t internalformat,
                                                 int *known);

uint32_t mglRenderMetalPixelFormatBytesPerPixel(uint32_t pixel_format);

uint32_t mglRenderMetalPixelFormatValueClass(uint32_t pixel_format);

int mglRenderPixelFormatIsDepthOrStencil(uint32_t pixel_format);

/* C1: DepthReadbackPlan -> mgl_readback_policy.h */
/* C1: DefaultDepthPixelFormat -> mgl_pso_format_class.h */
int mglRenderSamplerUnitExplicit(uint32_t flag);

int mglRenderPrefer1DSampler(uint32_t image_dim, int arrayed);

int mglRenderPrefer1DArraySampler(uint32_t image_dim, int arrayed);

int mglRenderTextureTargetIs1D(uint32_t target);

int mglRenderTextureTargetIsMSOr2DArray(uint32_t target);

void mglRenderMarkTextureLevelWritten(uint8_t *ever_written,
                                      uint8_t *has_initialized,
                                      uint8_t *suspicious_zero);

int mglRenderImageAccessWritable(uint32_t access);

uint32_t mglRenderSamplerObjectTarget(void);

int mglRenderPixelFormatIsUnorm8Color(uint32_t pixel_format);

int mglRenderTextureTargetIsCubeMap(uint32_t target);

int mglRenderTextureNeedsArrayLengthCheck(uint32_t target);

int mglRenderTextureTargetIsLayeredUpload(uint32_t target);

int mglRenderTextureTargetIs1DArray(uint32_t target);

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

int mglRenderTexturePrepareLevelUpload(
    const TextureLevel *level,
    uint32_t texture_type,
    uint32_t internal_format,
    uint32_t pixel_format,
    MGLRenderLevelUploadPrep *out);

/* RGB-family → RGBA expansion gates.  Pixel format is
 * the Apple MGLPixelFormat numeric value.  Returns 1/0. */
int mglRenderTextureInternalFormatNeedsRGBA8Expansion(
    uint32_t internal_format, uint32_t pixel_format);

int mglRenderTextureNeedsChannelExpansion(uint32_t internal_format,
                                             uint32_t pixel_format);

int mglRenderCreateSampler(void *sampler_descriptor,
                              void **sampler_out);

int mglRenderCreateDefaultSampler(void **sampler_out);

int mglRenderCreateFilterSampler(uint32_t nearest, void **sampler_out);

/* Translate GL texture parameters into a Metal-cpp sampler descriptor and
 * create the sampler without exposing MTL::* through this C ABI. */
/* Release the texture's Metal backing so the next bind recreates it (used when
 * the storage description changes, e.g. a renderbuffer's sample count). */
void mglRenderTextureDropBacking(Texture *texture);

/* FE-safe texture backend accessors (ARCH-007): no mgl_texture_backend.h. */
void *mglRenderTextureHandle(const Texture *texture);
void *mglRenderTextureSamplerHandle(const Texture *texture);
void *mglRenderTextureViewParentHandle(const Texture *texture);
void mglRenderTextureForgetHandle(Texture *texture);
void mglRenderTextureClearSamplerHandle(Texture *texture);
void mglRenderTextureClearViewParentHandle(Texture *texture);
void mglRenderTextureDropBaseLevelView(GLMContext ctx, Texture *texture);
void mglRenderTextureReleaseBackend(GLMContext ctx, Texture *texture);
int mglRenderTextureStorageIsFallback(const Texture *texture);
uint32_t mglRenderTextureWriteVersion(const Texture *texture);

int mglRenderCreateSamplerForGL(const TextureParameter *params,
                                   uint32_t target,
                                   void **sampler_out,
                                   char *err,
                                   size_t errcap);

int mglRenderBindingGetTextureSlotMask(MGLBindingState *binding_state, uint64_t mask_out[2]);

int mglRenderBindingSetTexture(MGLBindingState *binding_state, void *render_encoder, void *texture, uint32_t stage, uint32_t index);

int mglRenderBindingSetSampler(MGLBindingState *binding_state, void *render_encoder, void *sampler, uint32_t stage, uint32_t index);

int mglRenderBindingSetTextureForOwner(MGLBindingState *binding_state, MGLRenderEncoderOwner *render_encoder_owner, void *texture, uint32_t stage, uint32_t index);

int mglRenderBindingSetSamplerForOwner(MGLBindingState *binding_state, MGLRenderEncoderOwner *render_encoder_owner, void *sampler, uint32_t stage, uint32_t index);

int mglRenderBindingGetTexture(MGLBindingState *binding_state, uint32_t stage, uint32_t index, void **texture_out);

int mglRenderBindingGetSampler(MGLBindingState *binding_state, uint32_t stage, uint32_t index, void **sampler_out);

int mglRenderSetComputeTexture(void *compute_encoder,
                                  void *texture,
                                  uint32_t index);

int mglRenderSetComputeSampler(void *compute_encoder,
                                  void *sampler,
                                  uint32_t index);

int mglRenderTextureSampleParams(uint32_t target, int32_t samples,
                                 uint32_t *num_samples,
                                 uint32_t *sample_buffers);

void *mglRenderGetRenderPassAttachmentTextureOwner(MGLRenderPassStateOwner *owner, uint32_t attachment_kind, uint32_t color_index);

/* Encode a complete texture-to-texture preservation copy inside the owner.
 * Every common array slice and mip level is copied at its full mip extent;
 * encoder creation and endEncoding remain entirely in C++. */
int mglRenderCopyMatchingTextureSubresourcesForCommandBufferOwner(MGLCommandBufferOwner *command_buffer_owner, void *source_texture, void *destination_texture);

int mglRenderBlitSynchronizeTexture(void *blit_encoder,
                                       void *texture,
                                       uint64_t slice,
                                       uint64_t level);

int mglRenderSetRenderTexture(void *render_encoder,
                                 void *texture,
                                 uint32_t stage,
                                 uint32_t index);

int mglRenderSetRenderSampler(void *render_encoder,
                                 void *sampler,
                                 uint32_t stage,
                                 uint32_t index);

int mglRenderSetRenderTextureForOwner(MGLRenderEncoderOwner *render_encoder_owner, void *texture, uint32_t stage, uint32_t index);

int mglRenderSetRenderSamplerForOwner(MGLRenderEncoderOwner *render_encoder_owner, void *sampler, uint32_t stage, uint32_t index);

#ifdef __cplusplus
}
#endif

#endif
