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
 * mgl_readback_policy.c — C1 / O4.1 strip from mgl_render.cpp.
 * IntegerReadback source/packed/classify + CPU convert, Y-flip rows,
 * depth pack / depth-readback plan, GetTexImagePlan, MSAA array stride
 * (CTS ReadbackPolicy). Pure C; pixel_format uses MGLPixelFormat ABI.
 * Metal MSAA resolve encode remains in mgl_render.cpp.
 */

#include "mgl_readback_policy.h"

#include "glcorearb.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int mglRenderConvertIntegerReadback(
    const MGLRenderIntegerReadbackConvertParams* p) {
    if (!p || !p->src || !p->dst || !p->component_map ||
        !p->packed_bit_widths || !p->packed_shifts) {
        return -1;
    }
    const uint32_t src_pixel_bytes =
        p->source_rgb10a2_uint ? 4u :
        p->source_component_count * p->source_component_bytes;
    for (uint32_t y = 0; y < p->copy_h; y++) {
        const uint8_t* srcRow = p->src + (uint64_t)y * p->src_bytes_per_row;
        uint64_t outputY = p->flip_y ? (p->dst_y + (p->copy_h - 1u - y))
                                    : (p->dst_y + y);
        uint8_t* dstRow = (uint8_t *)p->dst + outputY * p->dst_bytes_per_row;
        for (uint32_t x = 0; x < p->copy_w; x++) {
            const uint8_t* s = srcRow + x * src_pixel_bytes;
            uint8_t* d = dstRow + (p->dst_x + x) * p->dst_pixel_bytes;

            /* Extract source component values (up to 4). */
            uint32_t srcValues[4] = {0, 0, 0, 0};
            for (uint32_t sc = 0; sc < p->source_component_count && sc < 4u; sc++) {
                if (p->source_rgb10a2_uint) {
                    uint32_t packed = *(const uint32_t *)(const void *)s;
                    static const uint8_t rgb10a2_shifts[4] = {0u, 10u, 20u, 30u};
                    static const uint32_t rgb10a2_masks[4] = {0x3ffu, 0x3ffu, 0x3ffu, 0x3u};
                    srcValues[sc] = (packed >> rgb10a2_shifts[sc]) & rgb10a2_masks[sc];
                } else if (p->source_component_bytes == 1u) {
                    srcValues[sc] = p->source_signed
                        ? (uint32_t)(int32_t)*(const int8_t *)(const void *)(s + sc)
                        : (uint32_t)s[sc];
                } else if (p->source_component_bytes == 2u) {
                    srcValues[sc] = p->source_signed
                        ? (uint32_t)(int32_t)*(const int16_t *)(const void *)(s + sc * 2u)
                        : (uint32_t)*(const uint16_t *)(const void *)(s + sc * 2u);
                } else {
                    srcValues[sc] = *(const uint32_t *)(const void *)(s + sc * 4u);
                }
            }

            if (p->is_packed_type) {
                /* Pack values into the packed format.
                 * Per OpenGL spec, integer values are CLAMPED to the bit width, not masked. */
                uint32_t packed = 0u;
                for (uint32_t c = 0; c < p->output_components && c < 4u; c++) {
                    int srcIdx = (c < 4u) ? p->component_map[c] : -1;
                    uint32_t val = 0u;
                    if (srcIdx >= 0 && (uint32_t)srcIdx < p->source_component_count) {
                        val = srcValues[srcIdx];
                    } else if (c == 3u) {
                        val = 1u; /* missing alpha → 1 */
                    }
                    /* Clamp to bit width (not mask). */
                    uint32_t maxVal = (p->packed_bit_widths[c] >= 32u) ? 0xFFFFFFFFu : ((1u << p->packed_bit_widths[c]) - 1u);
                    if (val > maxVal) val = maxVal;
                    packed |= val << p->packed_shifts[c];
                }
                if (p->packed_output_bytes == 1u) {
                    d[0] = (uint8_t)packed;
                } else if (p->packed_output_bytes == 2u) {
                    ((uint16_t *)(void *)d)[0] = (uint16_t)packed;
                } else {
                    ((uint32_t *)(void *)d)[0] = packed;
                }
            } else {
                /* Non-packed: write each component individually.
                 * Per OpenGL spec, integer values are CLAMPED to the output type range. */
                for (uint32_t c = 0; c < p->output_components; c++) {
                    int srcIdx = (c < 4u) ? p->component_map[c] : -1;
                    uint32_t value = 0u;
                    if (srcIdx >= 0 && (uint32_t)srcIdx < p->source_component_count) {
                        value = srcValues[srcIdx];
                    } else if (c == 3u) {
                        value = 1u; /* missing alpha → 1 */
                    }
                    if (p->output_component_bytes == 1u) {
                        if (p->packed_type == GL_BYTE) {
                            /* Signed byte: clamp to [-128, 127].
                             * If source is unsigned, values > 127 must clamp
                             * to 127 (not wrap to negative via int32_t cast). */
                            if (p->source_signed) {
                                int32_t sv = (int32_t)value;
                                if (sv > 127) sv = 127;
                                if (sv < -128) sv = -128;
                                d[c] = (uint8_t)(int8_t)sv;
                            } else {
                                if (value > 127u) value = 127u;
                                d[c] = (uint8_t)value;
                            }
                        } else {
                            /* Unsigned byte: clamp to [0, 255] */
                            if (value > 255u) value = 255u;
                            d[c] = (uint8_t)value;
                        }
                    } else if (p->output_component_bytes == 2u) {
                        if (p->packed_type == GL_SHORT) {
                            /* Signed short: clamp to [-32768, 32767].
                             * See comment above re: unsigned source. */
                            if (p->source_signed) {
                                int32_t sv = (int32_t)value;
                                if (sv > 32767) sv = 32767;
                                if (sv < -32768) sv = -32768;
                                ((uint16_t *)(void *)d)[c] = (uint16_t)(int16_t)sv;
                            } else {
                                if (value > 32767u) value = 32767u;
                                ((uint16_t *)(void *)d)[c] = (uint16_t)value;
                            }
                        } else {
                            /* Unsigned short: clamp to [0, 65535] */
                            if (value > 65535u) value = 65535u;
                            ((uint16_t *)(void *)d)[c] = (uint16_t)value;
                        }
                    } else {
                        if (p->packed_type == GL_INT) {
                            /* Signed int: if source is unsigned, clamp to
                             * [0, INT32_MAX] to avoid wrap. */
                            if (p->source_signed) {
                                ((uint32_t *)(void *)d)[c] = value;
                            } else {
                                if (value > 0x7FFFFFFFu) value = 0x7FFFFFFFu;
                                ((uint32_t *)(void *)d)[c] = value;
                            }
                        } else {
                            /* Unsigned int: clamp to [0, 4294967295] */
                            ((uint32_t *)(void *)d)[c] = value;
                        }
                    }
                }
            }
        }
    }


    return 0;
}

int mglRenderIntegerReadbackSourceClassify(
    uint32_t pixel_format, MGLRenderIntegerReadbackSource* out) {
    if (!out) return -1;
    out->component_count = 0;
    out->component_bytes = 0;
    out->source_signed = 0;
    out->source_rgb10a2_uint = 0;
    out->recognized = 0;
    switch (pixel_format) {
        case 13u /* MGLPixelFormatR8Uint */:
            out->component_count = 1u; out->component_bytes = 1u;
            out->recognized = 1;
            break;
        case 14u /* MGLPixelFormatR8Sint */:
            out->component_count = 1u; out->component_bytes = 1u;
            out->source_signed = 1; out->recognized = 1;
            break;
        case 23u /* MGLPixelFormatR16Uint */:
            out->component_count = 1u; out->component_bytes = 2u;
            out->recognized = 1;
            break;
        case 24u /* MGLPixelFormatR16Sint */:
            out->component_count = 1u; out->component_bytes = 2u;
            out->source_signed = 1; out->recognized = 1;
            break;
        case 53u /* MGLPixelFormatR32Uint */:
            out->component_count = 1u; out->component_bytes = 4u;
            out->recognized = 1;
            break;
        case 54u /* MGLPixelFormatR32Sint */:
            out->component_count = 1u; out->component_bytes = 4u;
            out->source_signed = 1; out->recognized = 1;
            break;
        case 33u /* MGLPixelFormatRG8Uint */:
            out->component_count = 2u; out->component_bytes = 1u;
            out->recognized = 1;
            break;
        case 34u /* MGLPixelFormatRG8Sint */:
            out->component_count = 2u; out->component_bytes = 1u;
            out->source_signed = 1; out->recognized = 1;
            break;
        case 63u /* MGLPixelFormatRG16Uint */:
            out->component_count = 2u; out->component_bytes = 2u;
            out->recognized = 1;
            break;
        case 64u /* MGLPixelFormatRG16Sint */:
            out->component_count = 2u; out->component_bytes = 2u;
            out->source_signed = 1; out->recognized = 1;
            break;
        case 103u /* MGLPixelFormatRG32Uint */:
            out->component_count = 2u; out->component_bytes = 4u;
            out->recognized = 1;
            break;
        case 104u /* MGLPixelFormatRG32Sint */:
            out->component_count = 2u; out->component_bytes = 4u;
            out->source_signed = 1; out->recognized = 1;
            break;
        case 73u /* MGLPixelFormatRGBA8Uint */:
            out->component_count = 4u; out->component_bytes = 1u;
            out->recognized = 1;
            break;
        case 74u /* MGLPixelFormatRGBA8Sint */:
            out->component_count = 4u; out->component_bytes = 1u;
            out->source_signed = 1; out->recognized = 1;
            break;
        case 113u /* MGLPixelFormatRGBA16Uint */:
            out->component_count = 4u; out->component_bytes = 2u;
            out->recognized = 1;
            break;
        case 114u /* MGLPixelFormatRGBA16Sint */:
            out->component_count = 4u; out->component_bytes = 2u;
            out->source_signed = 1; out->recognized = 1;
            break;
        case 123u /* MGLPixelFormatRGBA32Uint */:
            out->component_count = 4u; out->component_bytes = 4u;
            out->recognized = 1;
            break;
        case 124u /* MGLPixelFormatRGBA32Sint */:
            out->component_count = 4u; out->component_bytes = 4u;
            out->source_signed = 1; out->recognized = 1;
            break;
        case 91u /* MGLPixelFormatRGB10A2Uint */:
            out->component_count = 4u; out->component_bytes = 4u;
            out->source_rgb10a2_uint = 1; out->recognized = 1;
            break;
        default:
            break;
    }
    return 0;
}

int mglRenderIntegerReadbackPackedTypeClassify(
    uint32_t packed_type, MGLRenderIntegerPackedType* out) {
    if (!out) return -1;
    out->is_packed = 0;
    for (int i = 0; i < 4; i++) {
        out->bit_widths[i] = 0;
        out->shifts[i] = 0;
    }
    out->output_bytes = 0;
    out->output_components = 0;
    switch (packed_type) {
        case 0x8032: /* GL_UNSIGNED_BYTE_3_3_2 */
            out->is_packed = 1;
            out->bit_widths[0] = 3; out->bit_widths[1] = 3;
            out->bit_widths[2] = 2; out->bit_widths[3] = 0;
            out->shifts[0] = 5; out->shifts[1] = 2;
            out->shifts[2] = 0; out->shifts[3] = 0;
            out->output_bytes = 1; out->output_components = 3;
            break;
        case 0x8362: /* GL_UNSIGNED_BYTE_2_3_3_REV */
            out->is_packed = 1;
            out->bit_widths[0] = 3; out->bit_widths[1] = 3;
            out->bit_widths[2] = 2; out->bit_widths[3] = 0;
            out->shifts[0] = 0; out->shifts[1] = 3;
            out->shifts[2] = 6; out->shifts[3] = 0;
            out->output_bytes = 1; out->output_components = 3;
            break;
        case 0x8363: /* GL_UNSIGNED_SHORT_5_6_5 */
            out->is_packed = 1;
            out->bit_widths[0] = 5; out->bit_widths[1] = 6;
            out->bit_widths[2] = 5; out->bit_widths[3] = 0;
            out->shifts[0] = 11; out->shifts[1] = 5;
            out->shifts[2] = 0; out->shifts[3] = 0;
            out->output_bytes = 2; out->output_components = 3;
            break;
        case 0x8364: /* GL_UNSIGNED_SHORT_5_6_5_REV */
            out->is_packed = 1;
            out->bit_widths[0] = 5; out->bit_widths[1] = 6;
            out->bit_widths[2] = 5; out->bit_widths[3] = 0;
            out->shifts[0] = 0; out->shifts[1] = 5;
            out->shifts[2] = 11; out->shifts[3] = 0;
            out->output_bytes = 2; out->output_components = 3;
            break;
        case 0x8033: /* GL_UNSIGNED_SHORT_4_4_4_4 */
            out->is_packed = 1;
            out->bit_widths[0] = 4; out->bit_widths[1] = 4;
            out->bit_widths[2] = 4; out->bit_widths[3] = 4;
            out->shifts[0] = 12; out->shifts[1] = 8;
            out->shifts[2] = 4; out->shifts[3] = 0;
            out->output_bytes = 2; out->output_components = 4;
            break;
        case 0x8365: /* GL_UNSIGNED_SHORT_4_4_4_4_REV */
            out->is_packed = 1;
            out->bit_widths[0] = 4; out->bit_widths[1] = 4;
            out->bit_widths[2] = 4; out->bit_widths[3] = 4;
            out->shifts[0] = 0; out->shifts[1] = 4;
            out->shifts[2] = 8; out->shifts[3] = 12;
            out->output_bytes = 2; out->output_components = 4;
            break;
        case 0x8034: /* GL_UNSIGNED_SHORT_5_5_5_1 */
            out->is_packed = 1;
            out->bit_widths[0] = 5; out->bit_widths[1] = 5;
            out->bit_widths[2] = 5; out->bit_widths[3] = 1;
            out->shifts[0] = 11; out->shifts[1] = 6;
            out->shifts[2] = 1; out->shifts[3] = 0;
            out->output_bytes = 2; out->output_components = 4;
            break;
        case 0x8366: /* GL_UNSIGNED_SHORT_1_5_5_5_REV */
            out->is_packed = 1;
            out->bit_widths[0] = 5; out->bit_widths[1] = 5;
            out->bit_widths[2] = 5; out->bit_widths[3] = 1;
            out->shifts[0] = 0; out->shifts[1] = 5;
            out->shifts[2] = 10; out->shifts[3] = 15;
            out->output_bytes = 2; out->output_components = 4;
            break;
        case 0x8035: /* GL_UNSIGNED_INT_8_8_8_8 */
            out->is_packed = 1;
            out->bit_widths[0] = 8; out->bit_widths[1] = 8;
            out->bit_widths[2] = 8; out->bit_widths[3] = 8;
            out->shifts[0] = 24; out->shifts[1] = 16;
            out->shifts[2] = 8; out->shifts[3] = 0;
            out->output_bytes = 4; out->output_components = 4;
            break;
        case 0x8367: /* GL_UNSIGNED_INT_8_8_8_8_REV */
            out->is_packed = 1;
            out->bit_widths[0] = 8; out->bit_widths[1] = 8;
            out->bit_widths[2] = 8; out->bit_widths[3] = 8;
            out->shifts[0] = 0; out->shifts[1] = 8;
            out->shifts[2] = 16; out->shifts[3] = 24;
            out->output_bytes = 4; out->output_components = 4;
            break;
        case 0x8036: /* GL_UNSIGNED_INT_10_10_10_2 */
            out->is_packed = 1;
            out->bit_widths[0] = 10; out->bit_widths[1] = 10;
            out->bit_widths[2] = 10; out->bit_widths[3] = 2;
            out->shifts[0] = 22; out->shifts[1] = 12;
            out->shifts[2] = 2; out->shifts[3] = 0;
            out->output_bytes = 4; out->output_components = 4;
            break;
        case 0x8368: /* GL_UNSIGNED_INT_2_10_10_10_REV */
            out->is_packed = 1;
            out->bit_widths[0] = 10; out->bit_widths[1] = 10;
            out->bit_widths[2] = 10; out->bit_widths[3] = 2;
            out->shifts[0] = 0; out->shifts[1] = 10;
            out->shifts[2] = 20; out->shifts[3] = 30;
            out->output_bytes = 4; out->output_components = 4;
            break;
        default:
            break;
    }
    return 0;
}

int mglRenderIntegerReadbackClassify(
    uint32_t pixel_format, uint32_t gl_format, uint32_t gl_type,
    MGLRenderIntegerReadbackClassify* out) {
    if (!out) return -1;
    out->source_is_integer_texture = 0;
    out->output_is_integer_format = 0;
    out->output_components = 0;
    out->component_map[0] = 0; out->component_map[1] = 1;
    out->component_map[2] = 2; out->component_map[3] = 3;
    out->output_component_bytes = 0;
    switch (pixel_format) {
        case 13u /* MGLPixelFormatR8Uint */:
        case 14u /* MGLPixelFormatR8Sint */:
        case 23u /* MGLPixelFormatR16Uint */:
        case 24u /* MGLPixelFormatR16Sint */:
        case 53u /* MGLPixelFormatR32Uint */:
        case 54u /* MGLPixelFormatR32Sint */:
        case 33u /* MGLPixelFormatRG8Uint */:
        case 34u /* MGLPixelFormatRG8Sint */:
        case 63u /* MGLPixelFormatRG16Uint */:
        case 64u /* MGLPixelFormatRG16Sint */:
        case 103u /* MGLPixelFormatRG32Uint */:
        case 104u /* MGLPixelFormatRG32Sint */:
        case 73u /* MGLPixelFormatRGBA8Uint */:
        case 74u /* MGLPixelFormatRGBA8Sint */:
        case 113u /* MGLPixelFormatRGBA16Uint */:
        case 114u /* MGLPixelFormatRGBA16Sint */:
        case 123u /* MGLPixelFormatRGBA32Uint */:
        case 124u /* MGLPixelFormatRGBA32Sint */:
        case 91u /* MGLPixelFormatRGB10A2Uint */:
            out->source_is_integer_texture = 1;
            break;
        default:
            break;
    }
    switch (gl_format) {
        case GL_RED_INTEGER:
        case GL_RG_INTEGER:
        case GL_RGB_INTEGER:
        case GL_BGR_INTEGER:
        case GL_RGBA_INTEGER:
        case GL_BGRA_INTEGER:
        case 0x8d95: /* GL_GREEN_INTEGER */
        case 0x8d96: /* GL_BLUE_INTEGER */
        case 0x8d97: /* GL_ALPHA_INTEGER */
            out->output_is_integer_format = 1;
            break;
        default:
            break;
    }
    if (out->source_is_integer_texture && out->output_is_integer_format) {
        switch (gl_format) {
            case GL_RED_INTEGER:
                out->output_components = 1u;
                out->component_map[0] = 0; out->component_map[1] = -1;
                out->component_map[2] = -1; out->component_map[3] = -1;
                break;
            case GL_RG_INTEGER:
                out->output_components = 2u;
                out->component_map[0] = 0; out->component_map[1] = 1;
                out->component_map[2] = -1; out->component_map[3] = -1;
                break;
            case GL_RGB_INTEGER:
                out->output_components = 3u;
                out->component_map[0] = 0; out->component_map[1] = 1;
                out->component_map[2] = 2; out->component_map[3] = -1;
                break;
            case GL_BGR_INTEGER:
                out->output_components = 3u;
                out->component_map[0] = 2; out->component_map[1] = 1;
                out->component_map[2] = 0; out->component_map[3] = -1;
                break;
            case GL_RGBA_INTEGER:
                out->output_components = 4u;
                out->component_map[0] = 0; out->component_map[1] = 1;
                out->component_map[2] = 2; out->component_map[3] = 3;
                break;
            case GL_BGRA_INTEGER:
                out->output_components = 4u;
                out->component_map[0] = 2; out->component_map[1] = 1;
                out->component_map[2] = 0; out->component_map[3] = 3;
                break;
            case 0x8d95:
                out->output_components = 1u;
                out->component_map[0] = 1; out->component_map[1] = -1;
                out->component_map[2] = -1; out->component_map[3] = -1;
                break;
            case 0x8d96:
                out->output_components = 1u;
                out->component_map[0] = 2; out->component_map[1] = -1;
                out->component_map[2] = -1; out->component_map[3] = -1;
                break;
            case 0x8d97:
                out->output_components = 1u;
                out->component_map[0] = 3; out->component_map[1] = -1;
                out->component_map[2] = -1; out->component_map[3] = -1;
                break;
            default:
                break;
        }
        out->output_component_bytes =
            (gl_type == GL_BYTE || gl_type == GL_UNSIGNED_BYTE) ? 1u :
            (gl_type == GL_SHORT || gl_type == GL_UNSIGNED_SHORT) ? 2u : 4u;
    }
    return 0;
}

void mglRenderCopyRows(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t row_bytes, uint64_t height, int flip_y) {
    if (!src || !dst || row_bytes == 0u || height == 0u) {
        return;
    }
    const uint8_t *src_bytes = (const uint8_t *)src;
    uint8_t *dst_bytes = (uint8_t *)dst;
    for (uint64_t y = 0; y < height; y++) {
        const uint8_t *src_row = src_bytes + (y * src_bytes_per_row);
        uint64_t dst_y = flip_y ? (height - 1u - y) : y;
        uint8_t *dst_row = dst_bytes + (dst_y * dst_bytes_per_row);
        memcpy(dst_row, src_row, (size_t)row_bytes);
    }
}

void mglRenderCopyDepthTextureBytesToFloat(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint64_t src_depth_bytes, int is_depth16, int flip_y) {
    if (!src || !dst || width == 0u || height == 0u ||
        src_depth_bytes == 0u) {
        return;
    }
    const uint8_t *src_bytes = (const uint8_t *)src;
    uint8_t *dst_bytes = (uint8_t *)dst;
    for (uint64_t y = 0; y < height; y++) {
        const uint8_t *src_row = src_bytes + (y * src_bytes_per_row);
        uint64_t dst_y = flip_y ? (height - 1u - y) : y;
        float *dst_row = (float *)(void *)(dst_bytes + (dst_y * dst_bytes_per_row));
        for (uint64_t x = 0; x < width; x++) {
            if (is_depth16) {
                uint16_t value = 0u;
                memcpy(&value, src_row + (x * src_depth_bytes),
                       sizeof(value));
                dst_row[x] = (float)value / 65535.0f;
            } else {
                memcpy(&dst_row[x], src_row + (x * src_depth_bytes),
                       sizeof(float));
            }
        }
    }
}

int mglRenderDepthReadbackPlan(uint32_t pixel_format, int *is_depth16,
                               int *is_packed_d32f_s8) {
    int d16 = pixel_format == 250u /* MGLPixelFormatDepth16Unorm */ ? 1 : 0;
    int ds = pixel_format == 260u /* MGLPixelFormatDepth32Float_Stencil8 */
                 ? 1
                 : 0;
    if (is_depth16) {
        *is_depth16 = d16;
    }
    if (is_packed_d32f_s8) {
        *is_packed_d32f_s8 = ds;
    }
    return d16 || ds ||
                   pixel_format == 252u /* MGLPixelFormatDepth32Float */
               ? 1
               : 0;
}

void *mglRenderTextureRepackDepthPlanes(const void *bytes,
                                        size_t bytes_per_image,
                                        size_t expected_bytes_per_image,
                                        size_t copy_depth) {
    if (!bytes || expected_bytes_per_image == 0 ||
        bytes_per_image < expected_bytes_per_image || copy_depth == 0) {
        return NULL;
    }
    if (expected_bytes_per_image > SIZE_MAX / copy_depth) {
        return NULL;
    }
    size_t packed_size = expected_bytes_per_image * copy_depth;
    void *packed = malloc(packed_size);
    if (!packed) {
        return NULL;
    }
    const uint8_t *src = (const uint8_t *)bytes;
    uint8_t *dst = (uint8_t *)packed;
    for (size_t z = 0; z < copy_depth; z++) {
        memcpy(dst + z * expected_bytes_per_image,
               src + z * bytes_per_image, expected_bytes_per_image);
    }
    return packed;
}

uint32_t mglRenderMSAAArrayLayerStride(int layered, uint32_t textarget) {
    return layered && textarget == GL_TEXTURE_2D_MULTISAMPLE_ARRAY ? 8u : 1u;
}

int mglRenderGetTexImagePlan(
    uint32_t pixel_format, uint32_t gl_format, uint32_t gl_type,
    uint32_t width, uint32_t height, uint32_t depth,
    uint32_t dst_pixel_bytes, uint32_t source_bpp,
    int bgra8_format_compatible,
    uint32_t bytes_per_row, uint32_t bytes_per_image,
    int storage_private,
    MGLRenderGetTexImagePlan *out) {
    if (!out) return -1;
    out->direct_r32_float_read =
        (pixel_format == 55u /* MGLPixelFormatR32Float */ &&
         gl_format == GL_RED && gl_type == GL_FLOAT) ? 1 : 0;
    out->use_bgra8_conversion =
        (dst_pixel_bytes > 0u && depth == 1u &&
         !out->direct_r32_float_read && bgra8_format_compatible) ? 1 : 0;
    out->source_is_bgra8 =
        (pixel_format == 80u /* MGLPixelFormatBGRA8Unorm */ ||
         pixel_format == 81u /* MGLPixelFormatBGRA8Unorm_sRGB */ ||
         pixel_format == 70u /* MGLPixelFormatRGBA8Unorm */ ||
         pixel_format == 71u /* MGLPixelFormatRGBA8Unorm_sRGB */) ? 1 : 0;
    uint64_t row_bytes;
    if (out->use_bgra8_conversion && !out->source_is_bgra8 &&
        source_bpp > 0u) {
        row_bytes = (uint64_t)width * (uint64_t)source_bpp;
    } else if (out->use_bgra8_conversion) {
        row_bytes = (uint64_t)width * 4u;
    } else {
        row_bytes = bytes_per_row > 0
            ? (uint64_t)bytes_per_row
            : (uint64_t)width * (dst_pixel_bytes > 0
                                     ? (uint64_t)dst_pixel_bytes : 1u);
    }
    out->row_bytes = row_bytes;
    out->image_bytes = row_bytes * (uint64_t)height;
    out->total_bytes = out->image_bytes;
    if (!out->use_bgra8_conversion && storage_private &&
        bytes_per_image > 0 && depth > 1) {
        out->total_bytes = (uint64_t)bytes_per_image * (uint64_t)depth;
    }
    return 0;
}
