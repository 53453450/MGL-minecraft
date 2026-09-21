/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef MGL_RENDER_API_PIXEL_CONVERT_H
#define MGL_RENDER_API_PIXEL_CONVERT_H

/* Declarations for the pixel_convert slice of the renderer facade.
 * Value layouts live in mgl_render.h. Standalone include pulls
 * mgl_render_fwd.h (incomplete types) instead of the full facade. */

#ifndef MGL_RENDER_H
#include "mgl_render_fwd.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

int mglRenderPixelFormatIsIntegerColor(uint32_t pixel_format);

int mglRenderPixelFormatIsSignedIntegerColor(uint32_t pixel_format);

/* BGRA8/RGBA8 UNORM texture bytes -> GL channel
 * swizzle tail (UNSIGNED_BYTE, plus the leftover RGBA FLOAT branch).
 * Mirrors the ObjC final format switch (1 on success, 0 on bad args /
 * unsupported). */
int mglRenderCopyUnorm8SwizzleTextureBytesToGL(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y);

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

uint32_t mglRenderResolveUploadSwizzlePixelFormat(
    uint32_t native, int single_ch, uint32_t single_fmt, int int_multi,
    uint32_t int_fmt, int stencil, uint32_t stencil_fmt, int ds_depth,
    uint32_t ds_fmt);

uint32_t mglRenderIntegerFormatComponentMap(uint32_t format, int map[4]);

uint32_t mglRenderIntegerTypeComponentBytes(uint32_t type);

int mglRenderDepth32FStencil8NeedsUnpack(uint32_t internalformat,
                                         uint32_t pixel_format,
                                         uint32_t src_bpr, uint32_t width);

int mglRenderDirectR32FloatRead(uint32_t pixel_format, uint32_t format,
                                uint32_t type);

int mglRenderPixelFormatIsDepth32FloatStencil8(uint32_t pixel_format);

const char *mglRenderGLSLColumnSwizzle(uint32_t rows);

const char *mglRenderGLSLTypeSwizzle(uint32_t type);

const char *mglRenderGLSLIntegerAsFloatType(uint32_t type);

uint32_t mglRenderStructPackSrcStride(uint32_t src_stride, uint32_t elem_stride);

int mglRenderStructPackUseBulk(int32_t ai, int64_t buf_size, uint32_t member_size,
                               uint32_t src_stride);

int mglRenderShouldPackPlainUniformStruct(int spvc_type, int has_members,
                                          uint32_t member_count,
                                          uint64_t required, int sampler_like);

void mglRenderPackCurrentAttribPool(const uint8_t *values, uint32_t attrib_count,
                                    uint8_t *dst, uint64_t dst_bytes,
                                    uint32_t repeat_count, uint32_t value_bytes);

int mglRenderRGBExpandParams(uint32_t pixel_format, uint32_t *src_comp_bytes,
                             uint32_t *dst_comp_bytes, uint64_t *alpha_default);

/* R-only upload-swizzle gate.  swizzled==0 → 0;
 * otherwise the GL_R* internal-format table.  Returns 1/0. */
int mglRenderTextureUploadNeedsSingleChannelSwizzle(uint32_t internal_format,
                                                       int swizzled);

int mglRenderTextureUploadNeedsSingleChannelSwizzleBake(
    uint32_t internal_format, int swizzled);

/* Metal pixel format for single-channel swizzle upload expansion.
 * Returns MTLPixelFormatInvalid when the format is not handled. */
uint32_t mglRenderSingleChannelSwizzleStoragePixelFormat(
    uint32_t internal_format);

/* Multi-channel integer formats bake swizzle into CPU texels instead of
 * relying on Metal view swizzle (unreliable for Sint on some paths). */
int mglRenderTextureUploadNeedsIntegerMultiChannelSwizzleBake(
    uint32_t internal_format, int swizzled);

uint32_t mglRenderIntegerMultiChannelSwizzleStoragePixelFormat(
    uint32_t internal_format);

int mglRenderTextureUploadNeedsStencilSwizzleBake(
    uint32_t internal_format, int swizzled, uint32_t depth_stencil_mode);

int mglRenderTextureUploadNeedsDepthStencilDepthSwizzleBake(
    uint32_t internal_format, int swizzled, uint32_t depth_stencil_mode);

uint32_t mglRenderStencilSwizzleStoragePixelFormat(void);

/* Returns 1 when swizzle was baked at upload for this storage format. */
int mglRenderTextureSwizzleUsesUploadBake(
    uint32_t internal_format, int swizzled,
    uint32_t storage_pixel_format);

/* GL swizzle enum → Metal TextureSwizzle ABI value
 * (uint32_t).  components gates missing channels to Zero / One(for Alpha). */
uint32_t mglRenderMTLSwizzleForGLSwizzle(uint32_t gl_swizzle,
                                            uint32_t components);

uint8_t *mglRenderCreateIntegerMultiChannelSwizzledUpload(
    uint32_t internal_format,
    uint32_t swizzle_r, uint32_t swizzle_g,
    uint32_t swizzle_b, uint32_t swizzle_a,
    const void *src_data, size_t width, size_t height,
    size_t src_bytes_per_row,
    size_t *out_bytes_per_row, size_t *out_bytes_per_image);

#ifdef __cplusplus
}
#endif

#endif
