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
 * mgl_integer_readback.h
 *
 * C1 domain strip from mgl_render.cpp: integer texture readback
 * classification + CPU conversion. Pure data transforms / tables —
 * no Metal-cpp, no renderer instance. pixel_format arguments use the
 * MGLPixelFormat / MTL::PixelFormat numeric ABI (see pixel_utils.h).
 *
 * Callers historically went through mgl_render.h; that header now
 * includes this one so Texture.m keeps its existing call sites.
 */

#ifndef MGL_INTEGER_READBACK_H
#define MGL_INTEGER_READBACK_H

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

#ifdef __cplusplus
}
#endif

#endif /* MGL_INTEGER_READBACK_H */
