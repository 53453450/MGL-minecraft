/* SPDX-License-Identifier: LGPL-3.0-only */
/* Per-format RGBA8 expansion used by the sRGB/bake upload path.
 *
 * This is a layout table, not a format lookup: each internal format reads
 * different bits with its own alpha default, so the switch lives here in the
 * format layer instead of in the Metal lifecycle backend. */
#include "glcorearb.h"
#include "mgl_pixel_format.h"
#include "mgl_texture_compat.h"
#include "mgl_rgba8_expand.h"

/* Local copy: this file must link into the metalcpp smoke target, which does
 * not carry mgl_render_pixel.cpp. */
static uint32_t mglPackedUploadLE(const uint8_t *src, uint32_t bytes)
{
    uint32_t v = 0u;
    if (!src) return 0u;
    for (uint32_t i = 0u; i < bytes && i < 4u; i++) {
        v |= (uint32_t)src[i] << (8u * i);
    }
    return v;
}

static uint8_t mglExpandUNormTo8(uint32_t value, uint32_t bits)
{
    const uint32_t maxv = (1u << bits) - 1u;
    return (uint8_t)((value * 255u + maxv / 2u) / maxv);
}

void mglExpandInternalFormatToRGBA8(uint32_t internal_format,
                                    const uint8_t *src_pixel,
                                    uint32_t src_pixel_bytes,
                                    uint8_t out[4])
{
    uint32_t packed = mglPackedUploadLE(src_pixel, src_pixel_bytes);
    uint8_t r = 0u, g = 0u, b = 0u, a = 0xffu;
    switch (internal_format) {
        case GL_RGB8:
        case GL_SRGB8:
        case GL_RGB:
            r = src_pixel[0];
            g = src_pixel[1];
            b = src_pixel[2];
            a = 0xffu;
            break;
        case GL_RGB8_SNORM:
            r = src_pixel[0];
            g = src_pixel[1];
            b = src_pixel[2];
            a = 0x7fu; /* 1.0 in snorm */
            break;
        case GL_RGB8I:
        case GL_RGB8UI:
            r = src_pixel[0];
            g = src_pixel[1];
            b = src_pixel[2];
            a = 1u; /* 1 in integer */
            break;
        case GL_R3_G3_B2:
            r = mglExpandUNormTo8((packed >> 5u) & 0x7u, 3u);
            g = mglExpandUNormTo8((packed >> 2u) & 0x7u, 3u);
            b = mglExpandUNormTo8(packed & 0x3u, 2u);
            break;
        case GL_RGB4:
        case GL_RGB5:
        case GL_RGB565:
            r = mglExpandUNormTo8((packed >> 11u) & 0x1fu, 5u);
            g = mglExpandUNormTo8((packed >> 5u) & 0x3fu, 6u);
            b = mglExpandUNormTo8(packed & 0x1fu, 5u);
            break;
        case GL_RGB10:
            r = mglExpandUNormTo8(packed & 0x3ffu, 10u);
            g = mglExpandUNormTo8((packed >> 10u) & 0x3ffu, 10u);
            b = mglExpandUNormTo8((packed >> 20u) & 0x3ffu, 10u);
            break;
        case GL_RGB12:
            r = mglExpandUNormTo8(packed & 0xfffu, 12u);
            g = mglExpandUNormTo8((packed >> 12u) & 0xfffu, 12u);
            b = mglExpandUNormTo8((packed >> 24u) & 0xfffu, 12u);
            break;
        case GL_RGBA2:
        case GL_RGBA4:
            r = mglExpandUNormTo8((packed >> 12u) & 0xfu, 4u);
            g = mglExpandUNormTo8((packed >> 8u) & 0xfu, 4u);
            b = mglExpandUNormTo8((packed >> 4u) & 0xfu, 4u);
            a = mglExpandUNormTo8(packed & 0xfu, 4u);
            break;
        case GL_RGB5_A1:
            r = mglExpandUNormTo8((packed >> 11u) & 0x1fu, 5u);
            g = mglExpandUNormTo8((packed >> 6u) & 0x1fu, 5u);
            b = mglExpandUNormTo8((packed >> 1u) & 0x1fu, 5u);
            a = (packed & 0x1u) ? 0xffu : 0x00u;
            break;
                default:
            break;
            }
    out[0] = r;
    out[1] = g;
    out[2] = b;
    out[3] = a;
}
