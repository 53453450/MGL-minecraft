/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef MGL_RENDER_API_TEXTURE_H
#define MGL_RENDER_API_TEXTURE_H

/* Declarations for the texture slice of the renderer facade.
 * Value layouts live in mgl_render.h. Standalone include pulls
 * mgl_render_fwd.h (incomplete types) instead of the full facade. */

#ifndef MGL_RENDER_H
#include "mgl_render_fwd.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* C++-owned transient upload buffer. The returned buffer is borrowed from
 * the opaque owner; Metal command encoders retain it when a copy command is
 * recorded, so the owner may be destroyed immediately after encoding. */
int mglRenderCreateTextureStagingOwner(const void *bytes, uint64_t length, uint64_t resource_options, MGLTextureStagingOwner **owner_out, void **buffer_out);

void mglRenderDestroyTextureStagingOwner(MGLTextureStagingOwner **owner);

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
/* O3.3: sampled/storage/depth-recover/Y-flip/sampler-materialize -> mgl_binding_texture.h */
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

int mglRenderPassUsesColorTextureOwner(MGLRenderPassStateOwner *owner, void *texture, uint32_t *attachment_index_out);

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
