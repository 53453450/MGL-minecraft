/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/*
 * mgl_readback_policy.h
 *
 * C1 / O4.1 domain strip from mgl_render.cpp — IntegerReadback classify +
 * CPU convert, plus Y-flip / depth pack / GetTexImagePlan / MSAA array
 * stride policy (CTS ReadbackPolicy alignment). Pure data transforms /
 * tables; no Metal-cpp, no renderer instance. pixel_format uses the
 * MGLPixelFormat / MTL::PixelFormat numeric ABI (see pixel_utils.h).
 *
 * Metal MSAA resolve encode stays in mgl_render.cpp (residual).
 * Do not sink these helpers back into mgl_render.cpp.
 *
 * Callers historically went through mgl_render.h; that header includes
 * this one so Texture.m keeps its existing call sites.
 */

#ifndef MGL_READBACK_POLICY_H
#define MGL_READBACK_POLICY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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
    /* Single-sample RT Metal storage is top-row-first; flip to GL bottom-up. */
    int flip_y;
} MGLRenderIntegerReadbackConvertParams;

/* integer texture readback CPU conversion — the
 * per-pixel component extraction + GL_INTEGER packing/clamping loop of
 * mglReadIntegerTextureAsRGBA32, as a pure data transformation shared by
 * both gates.  Returns 0 on success, -1 on bad args. */
int mglRenderConvertIntegerReadback(
    const MGLRenderIntegerReadbackConvertParams *params);

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

/* --- C1 / O4.1 residual: Y-flip / depth pack / GetTexImagePlan / MSAA stride --- */

/* copy packed rows with optional Y-flip. Pure CPU memcpy of `row_bytes`
 * per row — mirrors mglMetalCopyRows (void). */
void mglRenderCopyRows(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t row_bytes, uint64_t height, int flip_y);

/* Depth16Unorm / unpacked depth-float rows -> GL float rows with optional
 * Y-flip. Mirrors the CPU convert loop in mglReadDepthTextureAsFloat
 * (void; bad args are a no-op). */
void mglRenderCopyDepthTextureBytesToFloat(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint64_t src_depth_bytes, int is_depth16, int flip_y);

/* Depth readback format plan — Depth16Unorm / Depth32Float /
 * Depth32Float_Stencil8 classification for the float readback path.
 * Pure numeric MGLPixelFormat ABI. Sets *is_depth16 / *is_packed_d32f_s8
 * when non-NULL. Returns 1 if the format is a supported depth readback
 * source, else 0. */
int mglRenderDepthReadbackPlan(uint32_t pixel_format, int *is_depth16,
                               int *is_packed_d32f_s8);

/* Repack strided 3D depth planes into the tight image stride required by
 * replaceRegion. Returns a malloc-owned buffer or NULL on invalid input
 * or allocation failure. */
void *mglRenderTextureRepackDepthPlanes(const void *bytes,
                                        size_t bytes_per_image,
                                        size_t expected_bytes_per_image,
                                        size_t copy_depth);

/* MSAA array layer stride policy — layered 2D_MULTISAMPLE_ARRAY uses
 * stride 8, otherwise 1. Pure classification (Metal encode of MSAA
 * resolve remains in mgl_render.cpp). */
uint32_t mglRenderMSAAArrayLayerStride(int layered, uint32_t textarget);

typedef struct MGLRenderGetTexImagePlan_t {
    int direct_r32_float_read;
    int use_bgra8_conversion;
    int source_is_bgra8;
    uint64_t row_bytes;
    uint64_t image_bytes;
    uint64_t total_bytes;
} MGLRenderGetTexImagePlan;

/* mtlGetTexImage staging plan — direct R32F read detection, BGRA8
 * conversion eligibility, source-is-BGRA8-family check, and
 * row/image/total byte computation. Shared by both gates; caller
 * resolves sizeForFormatType / readback bpp / format compatibility
 * through existing C helpers. pixel_format is MGLPixelFormat ABI. */
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

#ifdef __cplusplus
}
#endif

#endif /* MGL_READBACK_POLICY_H */
