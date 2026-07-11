/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * mgl_pixel_format.c
 * MGL — Pure pixel format conversion functions (no GLMContext dependency)
 */

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "mgl_pixel_format.h"
#include "pixel_utils.h"

extern GLint mglTexLevelCanonicalInternalFormat(GLint internalformat);
extern bool mglTexLevelInternalFormatCompressed(GLint internalformat);
extern GLint mglCompressedInternalFormatToSizedUncompressed(GLint internalformat);
extern GLint mglTexLevelComponentBits(GLint internalformat, GLenum pname);
extern GLint mglTexLevelComponentType(GLint internalformat, GLenum pname);

bool mglFindFirstNonZeroByte(const uint8_t *bytes, size_t len, size_t *offset_out, uint8_t *value_out)
{
    if (!bytes || len == 0) {
        return false;
    }

    for (size_t i = 0; i < len; i++) {
        if (bytes[i] != 0u) {
            if (offset_out) {
                *offset_out = i;
            }
            if (value_out) {
                *value_out = bytes[i];
            }
            return true;
        }
    }

    return false;
}

GLenum mglTextureComponentSizePname(GLuint component)
{
    switch (component) {
        case 0: return GL_TEXTURE_RED_SIZE;
        case 1: return GL_TEXTURE_GREEN_SIZE;
        case 2: return GL_TEXTURE_BLUE_SIZE;
        case 3: return GL_TEXTURE_ALPHA_SIZE;
        default: return GL_NONE;
    }
}

GLenum mglTextureComponentTypePname(GLuint component)
{
    switch (component) {
        case 0: return GL_TEXTURE_RED_TYPE;
        case 1: return GL_TEXTURE_GREEN_TYPE;
        case 2: return GL_TEXTURE_BLUE_TYPE;
        case 3: return GL_TEXTURE_ALPHA_TYPE;
        default: return GL_NONE;
    }
}

bool mglPackedCPUPixelLayoutForInternalFormat(GLenum internalformat,
                                                     size_t storage_pixel_size,
                                                     MGLCPUPixelLayout *layout)
{
    if (!layout || storage_pixel_size == 0u || storage_pixel_size > sizeof(uint64_t)) {
        return false;
    }

    size_t expected_size = sizeForInternalFormat(internalformat, 0, 0);
    if (expected_size != storage_pixel_size) {
        return false;
    }

    /* GL_RGB10_A2 / GL_RGB10_A2UI / GL_RGB10: Metal's RGB10A2 pixel format
     * uses the same LSB-first bit layout as GL_UNSIGNED_INT_2_10_10_10_REV.
     * GL_RGB10 has no alpha in the GL format but Metal stores it as RGB10A2,
     * so include the 2-bit alpha in the layout (defaults to 1.0 on upload
     * via mglReadExternalComponent, skipped on download for RGB formats).
     * Build a LSB-first packed layout so CPU conversion produces the correct
     * bit pattern for Metal. */
    if (internalformat == GL_RGB10_A2 || internalformat == GL_RGB10_A2UI ||
        internalformat == GL_RGB10) {
        if (storage_pixel_size != 4u) {
            return false;
        }
        memset(layout, 0, sizeof(*layout));
        layout->pixel_size = 4u;
        layout->packed = true;
        /* LSB-first: R[0:9], G[10:19], B[20:29], A[30:31] */
        GLuint bits[4] = {10u, 10u, 10u, 2u};
        GLuint bit_offset = 0u;
        for (GLuint i = 0u; i < 4u; i++) {
            layout->components[i].type =
                (internalformat == GL_RGB10_A2UI) ? GL_UNSIGNED_INT : GL_UNSIGNED_NORMALIZED;
            layout->components[i].bits = bits[i];
            layout->components[i].offset = 0u;
            layout->components[i].bit_offset = bit_offset;
            layout->components[i].storage_size = 4u;
            bit_offset += bits[i];
        }
        layout->component_count = 4u;
        return true;
    }

    /* GL_RGB4 / GL_RGB5: Metal upconverts these to RGBA8Unorm, but the CPU
     * shadow data is stored as GL_UNSIGNED_SHORT_5_6_5 (5+6+5 = 16 bits,
     * no unused bits).  The nominal GL bit counts (4+4+4=12 or 5+5+5=15)
     * leave unused bits which cause the generic packed path to reject them.
     * Use the actual 5_6_5 storage layout so CPU conversion matches what
     * mglCreateRGBA8ExpandedUpload expects.  MSB-first: R[11:15], G[5:10],
     * B[0:4]. */
    if (internalformat == GL_RGB4 || internalformat == GL_RGB5) {
        if (storage_pixel_size != 2u) {
            return false;
        }
        memset(layout, 0, sizeof(*layout));
        layout->pixel_size = 2u;
        layout->packed = true;
        GLuint bits[3] = {5u, 6u, 5u};
        GLuint bit_offset = 16u;  /* MSB-first */
        for (GLuint i = 0u; i < 3u; i++) {
            bit_offset -= bits[i];
            layout->components[i].type = GL_UNSIGNED_NORMALIZED;
            layout->components[i].bits = bits[i];
            layout->components[i].offset = 0u;
            layout->components[i].bit_offset = bit_offset;
            layout->components[i].storage_size = 2u;
        }
        layout->component_count = 3u;
        return true;
    }

    /* GL_R11F_G11F_B10F: Metal's RG11B10Float uses the same LSB-first bit
     * layout as GL_UNSIGNED_INT_10F_11F_11F_REV.  R and G are 11-bit floats
     * (5-bit exp, 6-bit mantissa), B is a 10-bit float (5-bit exp, 5-bit
     * mantissa).  All are unsigned (no sign bit).  Build a LSB-first packed
     * layout with custom MGL_FLOAT11/MGL_FLOAT10 component types so the
     * correct float encoding is used during CPU conversion. */
    if (internalformat == GL_R11F_G11F_B10F) {
        if (storage_pixel_size != 4u) {
            return false;
        }
        memset(layout, 0, sizeof(*layout));
        layout->pixel_size = 4u;
        layout->packed = true;
        /* LSB-first: R[0:10], G[11:21], B[22:31] */
        GLuint bits[3] = {11u, 11u, 10u};
        GLenum types[3] = {MGL_FLOAT11, MGL_FLOAT11, MGL_FLOAT10};
        GLuint bit_offset = 0u;
        for (GLuint i = 0u; i < 3u; i++) {
            layout->components[i].type = types[i];
            layout->components[i].bits = bits[i];
            layout->components[i].offset = 0u;
            layout->components[i].bit_offset = bit_offset;
            layout->components[i].storage_size = 4u;
            bit_offset += bits[i];
        }
        layout->component_count = 3u;
        return true;
    }

    /* Only use the packed path for formats with at least one non-byte-aligned
     * component (e.g. RGB565, RGBA4, RGB10_A2, R11F_G11F_B10F).  Formats whose
     * components are all 8/16/32 bits (R8, RG16F, R16I, R32F, RGBA8, ...) are
     * handled by the original path in mglBuildCPUPixelLayout which preserves
     * the correct component type (INT, FLOAT, ...) instead of forcing
     * UNSIGNED_NORMALIZED. */
    bool has_non_byte_aligned = false;

    GLuint total_bits = storage_pixel_size * 8u;
    GLuint component_count = 0u;
    for (GLuint component = 0u; component < 4u; component++) {
        GLenum component_name = component == 0u ? GL_RED :
                                component == 1u ? GL_GREEN :
                                component == 2u ? GL_BLUE : GL_ALPHA;
        GLuint bits = bitcountForInternalFormat(internalformat, component_name);
        if (bits == 0u) {
            continue;
        }
        if (bits > 32u ||
            bits > total_bits ||
            component_count >= 4u) {
            return false;
        }
        if (bits != 8u && bits != 16u && bits != 32u) {
            has_non_byte_aligned = true;
        }
        total_bits -= bits;
        MGLCPUPixelComponent *dst = &layout->components[component_count++];
        dst->type = GL_UNSIGNED_NORMALIZED;
        dst->bits = bits;
        dst->offset = 0u;
        dst->bit_offset = total_bits;
        dst->storage_size = storage_pixel_size;
    }


    if (component_count == 0u ||
        total_bits != 0u ||
        !has_non_byte_aligned) {
        return false;
    }

    layout->component_count = component_count;
    layout->pixel_size = storage_pixel_size;
    layout->packed = true;
    return true;
}

bool mglBuildCPUPixelLayout(GLenum internalformat,
                                   size_t storage_pixel_size,
                                   MGLCPUPixelLayout *layout)
{
    if (!layout || storage_pixel_size == 0u || storage_pixel_size > 64u) {
        return false;
    }

    /* Compressed internalformats are stored uncompressed (see
     * mglCompressedInternalFormatToSizedUncompressed).  Map them to their
     * sized uncompressed equivalents so component bits/type lookup succeeds. */
    internalformat = (GLenum)mglCompressedInternalFormatToSizedUncompressed((GLint)internalformat);

    memset(layout, 0, sizeof(*layout));
    layout->pixel_size = storage_pixel_size;

    /* Formats that Metal upconverts to RGBA8Unorm.  These have nominal bit
     * widths (4/5/1 bits) but Metal stores them as 8-bit per component.
     * Build a layout matching the Metal storage (4 x 8-bit unorm) so the
     * CPU conversion correctly scales values between the external format
     * and the Metal storage.  The 4th component (alpha) defaults to 1.0
     * for RGB-only formats via mglReadExternalComponent. */
    if (storage_pixel_size == 4u &&
        (internalformat == GL_RGB4 ||
         internalformat == GL_RGB5 ||
         internalformat == GL_RGBA4 ||
         internalformat == GL_RGB5_A1 ||
         internalformat == GL_R3_G3_B2 ||
         internalformat == GL_RGBA2)) {
        for (GLuint i = 0u; i < 4u; i++) {
            MGLCPUPixelComponent *dst = &layout->components[i];
            dst->type = GL_UNSIGNED_NORMALIZED;
            dst->bits = 8u;
            dst->offset = i;
            dst->bit_offset = 0u;
            dst->storage_size = 1u;
        }
        layout->component_count = 4u;
        return true;
    }

    if (mglPackedCPUPixelLayoutForInternalFormat(internalformat, storage_pixel_size, layout)) {
        return true;
    }

    size_t offset = 0u;
    for (GLuint component = 0; component < 4u; component++) {
        GLenum size_pname = mglTextureComponentSizePname(component);
        GLenum type_pname = mglTextureComponentTypePname(component);
        GLint bits = mglTexLevelComponentBits((GLint)internalformat, size_pname);
        if (bits == 0) {
            continue;
        }
        if (bits != 8 && bits != 16 && bits != 32) {
            return false;
        }

        size_t bytes = (size_t)bits / 8u;
        if (layout->component_count >= 4u ||
            offset > storage_pixel_size ||
            bytes > storage_pixel_size - offset) {
            return false;
        }

        MGLCPUPixelComponent *dst = &layout->components[layout->component_count++];
        dst->type = (GLenum)mglTexLevelComponentType((GLint)internalformat, type_pname);
        dst->bits = (GLuint)bits;
        dst->offset = offset;
        dst->bit_offset = 0u;
        dst->storage_size = bytes;
        offset += bytes;
    }

    /* Metal has no RGB-only pixel formats, so RGB internal formats are
     * upconverted to RGBA.  When the GL internal format has no alpha but
     * the storage pixel size includes space for it, add an implicit alpha
     * component so the CPU buffer layout matches the Metal texture. */
    if (offset < storage_pixel_size && layout->component_count > 0u &&
        layout->component_count < 4u) {
        size_t remaining = storage_pixel_size - offset;
        if (remaining == 1u || remaining == 2u || remaining == 4u) {
            MGLCPUPixelComponent *dst = &layout->components[layout->component_count++];
            /* Use the same type as the first component (unorm/snorm/int/float). */
            dst->type = layout->components[0].type;
            dst->bits = (GLuint)(remaining * 8u);
            dst->offset = offset;
            dst->bit_offset = 0u;
            dst->storage_size = remaining;
            offset = storage_pixel_size;
        }
    }

    return layout->component_count > 0u && offset == storage_pixel_size;
}

bool mglExternalFormatIsInteger(GLenum format)
{
    switch (format) {
        case GL_RED_INTEGER:
        case GL_RG_INTEGER:
        case GL_RGB_INTEGER:
        case GL_BGR_INTEGER:
        case GL_RGBA_INTEGER:
        case GL_BGRA_INTEGER:
        case 0x8d95: /* GL_GREEN_INTEGER */
        case 0x8d96: /* GL_BLUE_INTEGER */
        case 0x8d97: /* GL_ALPHA_INTEGER */
            return true;
        default:
            return false;
    }
}

/*
 * Accepted pixel-transfer `format` values for the (format,type) unpack path of
 * glTex(Sub)Image*.  Mirrors the set in verifyInternalFormatAndFormatType plus
 * the colour-index/depth/stencil transfer formats.  Used to reject unknown
 * formats (e.g. the smallest positive int not in this set) with GL_INVALID_ENUM.
 */
bool mglIsValidPixelTransferFormat(GLenum format)
{
    switch (format) {
        case GL_RED:
        case GL_RG:
        case GL_RGB:
        case GL_BGR:
        case GL_RGBA:
        case GL_BGRA:
        case GL_RED_INTEGER:
        case GL_RG_INTEGER:
        case GL_RGB_INTEGER:
        case GL_BGR_INTEGER:
        case GL_RGBA_INTEGER:
        case GL_BGRA_INTEGER:
        case GL_STENCIL_INDEX:
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_STENCIL:
        /* Legacy (deprecated but still emitted by virglrenderer). */
        case 0x1906: /* GL_ALPHA */
        case 0x1909: /* GL_LUMINANCE */
        case 0x190a: /* GL_LUMINANCE_ALPHA */
        case 0x8000: /* GL_COLOR_INDEX (legacy) */
        case 0x8d96: /* GL_GREEN_INTEGER */
        case 0x8d97: /* GL_BLUE_INTEGER */
        case 0x8d9c: /* GL_LUMINANCE_INTEGER_EXT */
        case 0x8d9d: /* GL_LUMINANCE_ALPHA_INTEGER_EXT */
            return true;
        default:
            return false;
    }
}

/*
 * Accepted pixel-transfer `type` values for the (format,type) unpack path.
 * Matches the set the CTS enumerates as the valid type constants; unknown
 * values (e.g. the smallest positive int not in this set) are rejected with
 * GL_INVALID_ENUM before upload.
 */
bool mglIsValidPixelTransferType(GLenum type)
{
    switch (type) {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_HALF_FLOAT:
        case GL_FLOAT:
        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_24_8:
        case 0x8c3b: /* GL_UNSIGNED_INT_10F_11F_11F_REV */
        case 0x8c3e: /* GL_UNSIGNED_INT_5_9_9_9_REV */
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
            return true;
        default:
            return false;
    }
}

bool mglInternalFormatIsInteger(GLint internalformat)
{
    switch (internalformat) {
        case GL_R8I: case GL_R16I: case GL_R32I:
        case GL_RG8I: case GL_RG16I: case GL_RG32I:
        case GL_RGB8I: case GL_RGB16I: case GL_RGB32I:
        case GL_RGBA8I: case GL_RGBA16I: case GL_RGBA32I:
        case GL_R8UI: case GL_R16UI: case GL_R32UI:
        case GL_RG8UI: case GL_RG16UI: case GL_RG32UI:
        case GL_RGB8UI: case GL_RGB16UI: case GL_RGB32UI:
        case GL_RGBA8UI: case GL_RGBA16UI: case GL_RGBA32UI:
        case GL_RGB10_A2UI:
            return true;
        default:
            return false;
    }
}

bool mglInternalFormatIsDepthStencil(GLint internalformat)
{
    switch (internalformat) {
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
        case GL_DEPTH_STENCIL:
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
        case GL_STENCIL_INDEX:
        case GL_STENCIL_INDEX1:
        case GL_STENCIL_INDEX4:
        case GL_STENCIL_INDEX8:
        case GL_STENCIL_INDEX16:
            return true;
        default:
            return false;
    }
}

/* Returns true if the internal format is a combined depth-stencil format
 * (i.e. has both depth and stencil components).  Used by GetTexImage and
 * ReadPixels to validate GL_DEPTH_STENCIL read format. */
bool mglInternalFormatIsCombinedDepthStencil(GLint internalformat)
{
    switch (internalformat) {
        case GL_DEPTH_STENCIL:
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
            return true;
        default:
            return false;
    }
}

int mglExternalSourceIndexForComponent(GLenum format, GLuint component)
{
    switch (format) {
        case GL_RED:
        case GL_RED_INTEGER:
            return component == 0u ? 0 : -1;
        case GL_RG:
        case GL_RG_INTEGER:
            return component < 2u ? (int)component : -1;
        case GL_RGB:
        case GL_RGB_INTEGER:
            return component < 3u ? (int)component : -1;
        case GL_RGBA:
        case GL_RGBA_INTEGER:
            return component < 4u ? (int)component : -1;
        case GL_BGR:
        case GL_BGR_INTEGER:
            if (component == 0u) return 2;
            if (component == 1u) return 1;
            if (component == 2u) return 0;
            return -1;
        case GL_BGRA:
        case GL_BGRA_INTEGER:
            if (component == 0u) return 2;
            if (component == 1u) return 1;
            if (component == 2u) return 0;
            if (component == 3u) return 3;
            return -1;
        default:
            return -1;
    }
}

double mglClampDouble(double v, double lo, double hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

double mglUnsignedMaxForBits(GLuint bits)
{
    if (bits >= 32u) {
        return 4294967295.0;
    }
    return (double)((1u << bits) - 1u);
}

uint64_t mglReadUnsignedLE(const uint8_t *src, size_t bytes)
{
    uint64_t value = 0u;
    if (!src) {
        return 0u;
    }
    if (bytes > sizeof(value)) {
        bytes = sizeof(value);
    }
    for (size_t i = 0; i < bytes; i++) {
        value |= ((uint64_t)src[i]) << (i * 8u);
    }
    return value;
}

void mglWriteUnsignedLE(uint8_t *dst, size_t bytes, uint64_t value)
{
    if (!dst) {
        return;
    }
    if (bytes > sizeof(value)) {
        bytes = sizeof(value);
    }
    for (size_t i = 0; i < bytes; i++) {
        dst[i] = (uint8_t)((value >> (i * 8u)) & 0xffu);
    }
}

/* Swap bytes within each element of size element_size in a pixel buffer.
 * Mirrors the GL_UNPACK_SWAP_BYTES / GL_PACK_SWAP_BYTES semantics: each
 * multi-byte element datum has its bytes reversed. 1-byte elements are
 * left untouched. */
void mglSwapPixelBytes(uint8_t *pixel, size_t pixel_size, size_t element_size)
{
    if (!pixel || element_size <= 1u || pixel_size == 0u) {
        return;
    }
    size_t offset = 0u;
    while (offset + element_size <= pixel_size) {
        for (size_t i = 0u; i < element_size / 2u; i++) {
            uint8_t tmp = pixel[offset + i];
            pixel[offset + i] = pixel[offset + element_size - 1u - i];
            pixel[offset + element_size - 1u - i] = tmp;
        }
        offset += element_size;
    }
}

double mglSignedMaxForBits(GLuint bits)
{
    if (bits >= 32u) {
        return 2147483647.0;
    }
    return (double)((1u << (bits - 1u)) - 1u);
}

double mglReadExternalComponent(const uint8_t *src,
                                       GLenum type,
                                       int source_index,
                                       bool integer_format,
                                       GLuint component)
{
    if (source_index < 0) {
        return component == 3u ? 1.0 : 0.0;
    }
    if (!src) {
        return 0.0;
    }

    uint64_t packed = 0u;
    switch (type) {
        case GL_UNSIGNED_BYTE_3_3_2:
            packed = mglReadUnsignedLE(src, sizeof(uint8_t));
            if (component == 0u) { uint32_t v = (uint32_t)((packed >> 5u) & 0x7u); return integer_format ? (double)v : (double)v / 7.0; }
            if (component == 1u) { uint32_t v = (uint32_t)((packed >> 2u) & 0x7u); return integer_format ? (double)v : (double)v / 7.0; }
            if (component == 2u) { uint32_t v = (uint32_t)(packed & 0x3u); return integer_format ? (double)v : (double)v / 3.0; }
            return 1.0;
        case GL_UNSIGNED_BYTE_2_3_3_REV:
            packed = mglReadUnsignedLE(src, sizeof(uint8_t));
            if (component == 0u) { uint32_t v = (uint32_t)(packed & 0x7u); return integer_format ? (double)v : (double)v / 7.0; }
            if (component == 1u) { uint32_t v = (uint32_t)((packed >> 3u) & 0x7u); return integer_format ? (double)v : (double)v / 7.0; }
            if (component == 2u) { uint32_t v = (uint32_t)((packed >> 6u) & 0x3u); return integer_format ? (double)v : (double)v / 3.0; }
            return 1.0;
        case GL_UNSIGNED_SHORT_5_6_5:
            packed = mglReadUnsignedLE(src, sizeof(uint16_t));
            if (component == 0u) { uint32_t v = (uint32_t)((packed >> 11u) & 0x1fu); return integer_format ? (double)v : (double)v / 31.0; }
            if (component == 1u) { uint32_t v = (uint32_t)((packed >> 5u) & 0x3fu); return integer_format ? (double)v : (double)v / 63.0; }
            if (component == 2u) { uint32_t v = (uint32_t)(packed & 0x1fu); return integer_format ? (double)v : (double)v / 31.0; }
            return 1.0;
        case GL_UNSIGNED_SHORT_5_6_5_REV:
            packed = mglReadUnsignedLE(src, sizeof(uint16_t));
            if (component == 0u) { uint32_t v = (uint32_t)(packed & 0x1fu); return integer_format ? (double)v : (double)v / 31.0; }
            if (component == 1u) { uint32_t v = (uint32_t)((packed >> 5u) & 0x3fu); return integer_format ? (double)v : (double)v / 63.0; }
            if (component == 2u) { uint32_t v = (uint32_t)((packed >> 11u) & 0x1fu); return integer_format ? (double)v : (double)v / 31.0; }
            return 1.0;
        case GL_UNSIGNED_SHORT_4_4_4_4:
            packed = mglReadUnsignedLE(src, sizeof(uint16_t));
            if (component == 0u) { uint32_t v = (uint32_t)((packed >> 12u) & 0xfu); return integer_format ? (double)v : (double)v / 15.0; }
            if (component == 1u) { uint32_t v = (uint32_t)((packed >> 8u) & 0xfu); return integer_format ? (double)v : (double)v / 15.0; }
            if (component == 2u) { uint32_t v = (uint32_t)((packed >> 4u) & 0xfu); return integer_format ? (double)v : (double)v / 15.0; }
            if (component == 3u) { uint32_t v = (uint32_t)(packed & 0xfu); return integer_format ? (double)v : (double)v / 15.0; }
            return 0.0;
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
            packed = mglReadUnsignedLE(src, sizeof(uint16_t));
            if (component == 0u) { uint32_t v = (uint32_t)(packed & 0xfu); return integer_format ? (double)v : (double)v / 15.0; }
            if (component == 1u) { uint32_t v = (uint32_t)((packed >> 4u) & 0xfu); return integer_format ? (double)v : (double)v / 15.0; }
            if (component == 2u) { uint32_t v = (uint32_t)((packed >> 8u) & 0xfu); return integer_format ? (double)v : (double)v / 15.0; }
            if (component == 3u) { uint32_t v = (uint32_t)((packed >> 12u) & 0xfu); return integer_format ? (double)v : (double)v / 15.0; }
            return 0.0;
        case GL_UNSIGNED_SHORT_5_5_5_1:
            packed = mglReadUnsignedLE(src, sizeof(uint16_t));
            if (component == 0u) { uint32_t v = (uint32_t)((packed >> 11u) & 0x1fu); return integer_format ? (double)v : (double)v / 31.0; }
            if (component == 1u) { uint32_t v = (uint32_t)((packed >> 6u) & 0x1fu); return integer_format ? (double)v : (double)v / 31.0; }
            if (component == 2u) { uint32_t v = (uint32_t)((packed >> 1u) & 0x1fu); return integer_format ? (double)v : (double)v / 31.0; }
            if (component == 3u) { uint32_t v = (uint32_t)(packed & 0x1u); return integer_format ? (double)v : (double)v; }
            return 0.0;
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
            packed = mglReadUnsignedLE(src, sizeof(uint16_t));
            if (component == 0u) { uint32_t v = (uint32_t)(packed & 0x1fu); return integer_format ? (double)v : (double)v / 31.0; }
            if (component == 1u) { uint32_t v = (uint32_t)((packed >> 5u) & 0x1fu); return integer_format ? (double)v : (double)v / 31.0; }
            if (component == 2u) { uint32_t v = (uint32_t)((packed >> 10u) & 0x1fu); return integer_format ? (double)v : (double)v / 31.0; }
            if (component == 3u) { uint32_t v = (uint32_t)((packed >> 15u) & 0x1u); return integer_format ? (double)v : (double)v; }
            return 0.0;
        case GL_UNSIGNED_INT_8_8_8_8:
            packed = mglReadUnsignedLE(src, sizeof(uint32_t));
            if (component == 0u) { uint32_t v = (uint32_t)((packed >> 24u) & 0xffu); return integer_format ? (double)v : (double)v / 255.0; }
            if (component == 1u) { uint32_t v = (uint32_t)((packed >> 16u) & 0xffu); return integer_format ? (double)v : (double)v / 255.0; }
            if (component == 2u) { uint32_t v = (uint32_t)((packed >> 8u) & 0xffu); return integer_format ? (double)v : (double)v / 255.0; }
            if (component == 3u) { uint32_t v = (uint32_t)(packed & 0xffu); return integer_format ? (double)v : (double)v / 255.0; }
            return 0.0;
        case GL_UNSIGNED_INT_8_8_8_8_REV:
            packed = mglReadUnsignedLE(src, sizeof(uint32_t));
            if (component == 0u) { uint32_t v = (uint32_t)(packed & 0xffu); return integer_format ? (double)v : (double)v / 255.0; }
            if (component == 1u) { uint32_t v = (uint32_t)((packed >> 8u) & 0xffu); return integer_format ? (double)v : (double)v / 255.0; }
            if (component == 2u) { uint32_t v = (uint32_t)((packed >> 16u) & 0xffu); return integer_format ? (double)v : (double)v / 255.0; }
            if (component == 3u) { uint32_t v = (uint32_t)((packed >> 24u) & 0xffu); return integer_format ? (double)v : (double)v / 255.0; }
            return 0.0;
        case GL_UNSIGNED_INT_10_10_10_2:
            packed = mglReadUnsignedLE(src, sizeof(uint32_t));
            if (component == 0u) { uint32_t v = (uint32_t)((packed >> 22u) & 0x3ffu); return integer_format ? (double)v : (double)v / 1023.0; }
            if (component == 1u) { uint32_t v = (uint32_t)((packed >> 12u) & 0x3ffu); return integer_format ? (double)v : (double)v / 1023.0; }
            if (component == 2u) { uint32_t v = (uint32_t)((packed >> 2u) & 0x3ffu); return integer_format ? (double)v : (double)v / 1023.0; }
            if (component == 3u) { uint32_t v = (uint32_t)(packed & 0x3u); return integer_format ? (double)v : (double)v / 3.0; }
            return 0.0;
        case GL_UNSIGNED_INT_2_10_10_10_REV:
            packed = mglReadUnsignedLE(src, sizeof(uint32_t));
            if (component == 0u) { uint32_t v = (uint32_t)(packed & 0x3ffu); return integer_format ? (double)v : (double)v / 1023.0; }
            if (component == 1u) { uint32_t v = (uint32_t)((packed >> 10u) & 0x3ffu); return integer_format ? (double)v : (double)v / 1023.0; }
            if (component == 2u) { uint32_t v = (uint32_t)((packed >> 20u) & 0x3ffu); return integer_format ? (double)v : (double)v / 1023.0; }
            if (component == 3u) { uint32_t v = (uint32_t)((packed >> 30u) & 0x3u); return integer_format ? (double)v : (double)v / 3.0; }
            return 0.0;
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
            packed = mglReadUnsignedLE(src, sizeof(uint32_t));
            /* R11F_G11F_B10F is an unsigned float format that can represent
             * values up to ~65024.  Do NOT clamp to [0,1] — the clamping is
             * the responsibility of the destination format's store path. */
            if (component == 0u) return (double)mglUnpackUnsignedFloatComponent((uint32_t)(packed & 0x7ffu), 6u);
            if (component == 1u) return (double)mglUnpackUnsignedFloatComponent((uint32_t)((packed >> 11u) & 0x7ffu), 6u);
            if (component == 2u) return (double)mglUnpackUnsignedFloatComponent((uint32_t)((packed >> 22u) & 0x3ffu), 5u);
            return 1.0;
        case GL_UNSIGNED_INT_5_9_9_9_REV: {
            packed = mglReadUnsignedLE(src, sizeof(uint32_t));
            /* GL_UNSIGNED_INT_5_9_9_9_REV (GL_RGB9_E5) uses a shared exponent:
             * value = mantissa * 2^(exp - 24) for all exp including 0.
             * Delegate to the shared implementation to avoid divergence. */
            double r, g, b;
            mglUnpackSharedExp(packed, &r, &g, &b);
            if (component == 0u) return r;
            if (component == 1u) return g;
            if (component == 2u) return b;
            return 1.0;
        }
        default:
            break;
    }

    size_t component_size = sizeForType(type);
    if (component_size == 0u) {
        return 0.0;
    }
    const uint8_t *p = src + ((size_t)source_index * component_size);

    switch (type) {
        case GL_UNSIGNED_BYTE: {
            uint8_t v;
            memcpy(&v, p, sizeof(v));
            return integer_format ? (double)v : ((double)v / 255.0);
        }
        case GL_BYTE: {
            int8_t v;
            memcpy(&v, p, sizeof(v));
            return integer_format ? (double)v : mglClampDouble((double)v / 127.0, -1.0, 1.0);
        }
        case GL_UNSIGNED_SHORT: {
            uint16_t v;
            memcpy(&v, p, sizeof(v));
            return integer_format ? (double)v : ((double)v / 65535.0);
        }
        case GL_SHORT: {
            int16_t v;
            memcpy(&v, p, sizeof(v));
            return integer_format ? (double)v : mglClampDouble((double)v / 32767.0, -1.0, 1.0);
        }
        case GL_UNSIGNED_INT: {
            uint32_t v;
            memcpy(&v, p, sizeof(v));
            return integer_format ? (double)v : ((double)v / 4294967295.0);
        }
        case GL_INT: {
            int32_t v;
            memcpy(&v, p, sizeof(v));
            return integer_format ? (double)v : mglClampDouble((double)v / 2147483647.0, -1.0, 1.0);
        }
        case GL_FLOAT: {
            float v;
            memcpy(&v, p, sizeof(v));
            return (double)v;
        }
        case GL_HALF_FLOAT: {
            uint16_t v;
            memcpy(&v, p, sizeof(v));
            return (double)mglHalfToFloat(v);
        }
        default:
            return 0.0;
    }
}

/* Unpack an 11-bit unsigned float (UE11) to float.
 * 5-bit exponent (bias 15), 6-bit mantissa, no sign bit. */
float mglUE11ToFloat(uint32_t v)
{
    uint32_t exponent = (v >> 6u) & 31u;
    uint32_t mantissa = v & 63u;
    if (exponent == 0u) {
        if (mantissa == 0u) return 0.0f;
        return ldexpf((float)mantissa, -24);
    }
    if (exponent == 31u) {
        return mantissa ? NAN : INFINITY;
    }
    return ldexpf(1.0f + (float)mantissa / 64.0f, (int)exponent - 15);
}

/* Unpack a 10-bit unsigned float (UE10) to float.
 * 5-bit exponent (bias 15), 5-bit mantissa, no sign bit. */
float mglUE10ToFloat(uint32_t v)
{
    uint32_t exponent = (v >> 5u) & 31u;
    uint32_t mantissa = v & 31u;
    if (exponent == 0u) {
        if (mantissa == 0u) return 0.0f;
        return ldexpf((float)mantissa, -24);
    }
    if (exponent == 31u) {
        return mantissa ? NAN : INFINITY;
    }
    return ldexpf(1.0f + (float)mantissa / 32.0f, (int)exponent - 15);
}

void mglStoreInternalComponent(uint8_t *dst,
                                      const MGLCPUPixelComponent *component,
                                      double value)
{
    uint8_t *p = dst + component->offset;
    switch (component->type) {
        case GL_FLOAT: {
            float v = (float)value;
            memcpy(p, &v, sizeof(v));
            break;
        }
        case GL_HALF_FLOAT: {
            uint16_t v = mglFloatToHalf((float)value);
            memcpy(p, &v, sizeof(v));
            break;
        }
        case GL_UNSIGNED_NORMALIZED: {
            double maxv = mglUnsignedMaxForBits(component->bits);
            uint32_t v = (uint32_t)(mglClampDouble(value, 0.0, 1.0) * maxv + 0.5);
            if (component->storage_size > 0u &&
                component->storage_size <= sizeof(uint64_t) &&
                (component->bit_offset != 0u ||
                 component->bits != component->storage_size * 8u)) {
                uint64_t packed = mglReadUnsignedLE(p, component->storage_size);
                uint64_t mask = component->bits >= 64u ? UINT64_MAX : ((1ull << component->bits) - 1ull);
                packed &= ~(mask << component->bit_offset);
                packed |= (((uint64_t)v) & mask) << component->bit_offset;
                mglWriteUnsignedLE(p, component->storage_size, packed);
            } else if (component->bits == 8u) {
                uint8_t u8 = (uint8_t)v;
                memcpy(p, &u8, sizeof(u8));
            } else if (component->bits == 16u) {
                uint16_t u16 = (uint16_t)v;
                memcpy(p, &u16, sizeof(u16));
            } else {
                memcpy(p, &v, sizeof(v));
            }
            break;
        }
        case GL_SIGNED_NORMALIZED: {
            double maxv = mglSignedMaxForBits(component->bits);
            int32_t v = (int32_t)(mglClampDouble(value, -1.0, 1.0) * maxv + (value >= 0.0 ? 0.5 : -0.5));
            if (component->bits == 8u) {
                int8_t s8 = (int8_t)v;
                memcpy(p, &s8, sizeof(s8));
            } else if (component->bits == 16u) {
                int16_t s16 = (int16_t)v;
                memcpy(p, &s16, sizeof(s16));
            } else {
                memcpy(p, &v, sizeof(v));
            }
            break;
        }
        case GL_UNSIGNED_INT: {
            /* Clamp to the component's bit-width range, not the full 32-bit range.
             * Per OpenGL spec, integer values must be clamped to the destination
             * type range, not masked/truncated. */
            double uintMax = (component->bits >= 32u) ? 4294967295.0 :
                             (double)((1ull << component->bits) - 1ull);
            uint32_t v = (uint32_t)mglClampDouble(value, 0.0, uintMax);
            if (component->storage_size > 0u &&
                component->storage_size <= sizeof(uint64_t) &&
                (component->bit_offset != 0u ||
                 component->bits != component->storage_size * 8u)) {
                /* Packed layout: read-modify-write to preserve other components. */
                uint64_t packed = mglReadUnsignedLE(p, component->storage_size);
                uint64_t mask = component->bits >= 64u ? UINT64_MAX : ((1ull << component->bits) - 1ull);
                packed &= ~(mask << component->bit_offset);
                packed |= (((uint64_t)v) & mask) << component->bit_offset;
                mglWriteUnsignedLE(p, component->storage_size, packed);
            } else if (component->bits == 8u) {
                uint8_t u8 = (uint8_t)v;
                memcpy(p, &u8, sizeof(u8));
            } else if (component->bits == 16u) {
                uint16_t u16 = (uint16_t)v;
                memcpy(p, &u16, sizeof(u16));
            } else {
                memcpy(p, &v, sizeof(v));
            }
            break;
        }
        case GL_INT: {
            /* Clamp to the component's bit-width range, not the full 32-bit range.
             * Per OpenGL spec, integer values must be clamped to the destination
             * type range, not masked/truncated. */
            double intMax = (component->bits >= 32u) ? 2147483647.0 :
                            (double)((1ll << (component->bits - 1u)) - 1ll);
            double intMin = (component->bits >= 32u) ? -2147483648.0 :
                            -(double)(1ll << (component->bits - 1u));
            int32_t v = (int32_t)mglClampDouble(value, intMin, intMax);
            if (component->storage_size > 0u &&
                component->storage_size <= sizeof(uint64_t) &&
                (component->bit_offset != 0u ||
                 component->bits != component->storage_size * 8u)) {
                /* Packed layout: read-modify-write to preserve other components. */
                uint64_t packed = mglReadUnsignedLE(p, component->storage_size);
                uint64_t mask = component->bits >= 64u ? UINT64_MAX : ((1ull << component->bits) - 1ull);
                uint64_t uval = (uint64_t)(int64_t)v & mask;
                packed &= ~(mask << component->bit_offset);
                packed |= uval << component->bit_offset;
                mglWriteUnsignedLE(p, component->storage_size, packed);
            } else if (component->bits == 8u) {
                int8_t s8 = (int8_t)v;
                memcpy(p, &s8, sizeof(s8));
            } else if (component->bits == 16u) {
                int16_t s16 = (int16_t)v;
                memcpy(p, &s16, sizeof(s16));
            } else {
                memcpy(p, &v, sizeof(v));
            }
            break;
        }
        case MGL_FLOAT11: {
            uint32_t f11 = mglFloatToFloat11((float)value);
            uint64_t packed = mglReadUnsignedLE(p, component->storage_size);
            uint64_t mask = (1ull << component->bits) - 1ull;
            packed &= ~(mask << component->bit_offset);
            packed |= ((uint64_t)f11 & mask) << component->bit_offset;
            mglWriteUnsignedLE(p, component->storage_size, packed);
            break;
        }
        case MGL_FLOAT10: {
            uint32_t f10 = mglFloatToFloat10((float)value);
            uint64_t packed = mglReadUnsignedLE(p, component->storage_size);
            uint64_t mask = (1ull << component->bits) - 1ull;
            packed &= ~(mask << component->bit_offset);
            packed |= ((uint64_t)f10 & mask) << component->bit_offset;
            mglWriteUnsignedLE(p, component->storage_size, packed);
            break;
        }
        default:
            break;
    }
}

double mglLoadInternalComponent(const uint8_t *src,
                                       const MGLCPUPixelComponent *component)
{
    const uint8_t *p = src + component->offset;
    switch (component->type) {
        case GL_FLOAT: {
            float v;
            memcpy(&v, p, sizeof(v));
            return (double)v;
        }
        case GL_HALF_FLOAT: {
            uint16_t v;
            memcpy(&v, p, sizeof(v));
            return (double)mglHalfToFloat(v);
        }
        case GL_UNSIGNED_NORMALIZED: {
            uint32_t v = 0u;
            if (component->storage_size > 0u &&
                component->storage_size <= sizeof(uint64_t) &&
                (component->bit_offset != 0u ||
                 component->bits != component->storage_size * 8u)) {
                uint64_t packed = mglReadUnsignedLE(p, component->storage_size);
                uint64_t mask = component->bits >= 64u ? UINT64_MAX : ((1ull << component->bits) - 1ull);
                v = (uint32_t)((packed >> component->bit_offset) & mask);
            } else if (component->bits == 8u) {
                uint8_t u8;
                memcpy(&u8, p, sizeof(u8));
                v = u8;
            } else if (component->bits == 16u) {
                uint16_t u16;
                memcpy(&u16, p, sizeof(u16));
                v = u16;
            } else {
                memcpy(&v, p, sizeof(v));
            }
            return (double)v / mglUnsignedMaxForBits(component->bits);
        }
        case GL_SIGNED_NORMALIZED: {
            int32_t v = 0;
            if (component->bits == 8u) {
                int8_t s8;
                memcpy(&s8, p, sizeof(s8));
                v = s8;
            } else if (component->bits == 16u) {
                int16_t s16;
                memcpy(&s16, p, sizeof(s16));
                v = s16;
            } else {
                memcpy(&v, p, sizeof(v));
            }
            return mglClampDouble((double)v / mglSignedMaxForBits(component->bits), -1.0, 1.0);
        }
        case GL_UNSIGNED_INT: {
            uint32_t v = 0u;
            if (component->storage_size > 0u &&
                component->storage_size <= sizeof(uint64_t) &&
                (component->bit_offset != 0u ||
                 component->bits != component->storage_size * 8u)) {
                /* Packed layout: extract component from packed word. */
                uint64_t packed = mglReadUnsignedLE(p, component->storage_size);
                uint64_t mask = component->bits >= 64u ? UINT64_MAX : ((1ull << component->bits) - 1ull);
                v = (uint32_t)((packed >> component->bit_offset) & mask);
            } else if (component->bits == 8u) {
                uint8_t u8;
                memcpy(&u8, p, sizeof(u8));
                v = u8;
            } else if (component->bits == 16u) {
                uint16_t u16;
                memcpy(&u16, p, sizeof(u16));
                v = u16;
            } else {
                memcpy(&v, p, sizeof(v));
            }
            return (double)v;
        }
        case GL_INT: {
            int32_t v = 0;
            if (component->storage_size > 0u &&
                component->storage_size <= sizeof(uint64_t) &&
                (component->bit_offset != 0u ||
                 component->bits != component->storage_size * 8u)) {
                /* Packed layout: extract component from packed word. */
                uint64_t packed = mglReadUnsignedLE(p, component->storage_size);
                uint64_t mask = component->bits >= 64u ? UINT64_MAX : ((1ull << component->bits) - 1ull);
                uint32_t uval = (uint32_t)((packed >> component->bit_offset) & mask);
                /* Sign-extend from component->bits to 32 bits. */
                if (component->bits < 32u && (uval & (1u << (component->bits - 1u)))) {
                    uval |= 0xFFFFFFFFu << component->bits;
                }
                v = (int32_t)uval;
            } else if (component->bits == 8u) {
                int8_t s8;
                memcpy(&s8, p, sizeof(s8));
                v = s8;
            } else if (component->bits == 16u) {
                int16_t s16;
                memcpy(&s16, p, sizeof(s16));
                v = s16;
            } else {
                memcpy(&v, p, sizeof(v));
            }
            return (double)v;
        }
        case MGL_FLOAT11: {
            uint64_t packed = mglReadUnsignedLE(p, component->storage_size);
            uint64_t mask = (1ull << component->bits) - 1ull;
            uint32_t v = (uint32_t)((packed >> component->bit_offset) & mask);
            return (double)mglUE11ToFloat(v);
        }
        case MGL_FLOAT10: {
            uint64_t packed = mglReadUnsignedLE(p, component->storage_size);
            uint64_t mask = (1ull << component->bits) - 1ull;
            uint32_t v = (uint32_t)((packed >> component->bit_offset) & mask);
            return (double)mglUE10ToFloat(v);
        }
        default:
            return 0.0;
    }
}

void mglWriteExternalComponent(uint8_t *dst,
                                      GLenum type,
                                      int dest_index,
                                      bool integer_format,
                                      double value)
{
    if (dest_index < 0 || !dst) {
        return;
    }

    switch (type) {
        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV: {
            uint8_t bits = 0u;
            uint8_t shift = 0u;
            size_t storage_size = sizeForType(type);
            switch (type) {
                case GL_UNSIGNED_BYTE_3_3_2:
                    bits = dest_index == 0 ? 3u : dest_index == 1 ? 3u : 2u;
                    shift = dest_index == 0 ? 5u : dest_index == 1 ? 2u : 0u;
                    break;
                case GL_UNSIGNED_BYTE_2_3_3_REV:
                    bits = dest_index == 0 ? 3u : dest_index == 1 ? 3u : 2u;
                    shift = dest_index == 0 ? 0u : dest_index == 1 ? 3u : 6u;
                    break;
                case GL_UNSIGNED_SHORT_5_6_5:
                    bits = dest_index == 1 ? 6u : 5u;
                    shift = dest_index == 0 ? 11u : dest_index == 1 ? 5u : 0u;
                    break;
                case GL_UNSIGNED_SHORT_5_6_5_REV:
                    bits = dest_index == 1 ? 6u : 5u;
                    shift = dest_index == 0 ? 0u : dest_index == 1 ? 5u : 11u;
                    break;
                case GL_UNSIGNED_SHORT_4_4_4_4:
                    bits = 4u;
                    shift = (uint8_t)(12u - ((uint8_t)dest_index * 4u));
                    break;
                case GL_UNSIGNED_SHORT_4_4_4_4_REV:
                    bits = 4u;
                    shift = (uint8_t)((uint8_t)dest_index * 4u);
                    break;
                case GL_UNSIGNED_SHORT_5_5_5_1:
                    bits = dest_index == 3 ? 1u : 5u;
                    shift = dest_index == 0 ? 11u : dest_index == 1 ? 6u : dest_index == 2 ? 1u : 0u;
                    break;
                case GL_UNSIGNED_SHORT_1_5_5_5_REV:
                    bits = dest_index == 3 ? 1u : 5u;
                    shift = dest_index == 0 ? 0u : dest_index == 1 ? 5u : dest_index == 2 ? 10u : 15u;
                    break;
                case GL_UNSIGNED_INT_8_8_8_8:
                    bits = 8u;
                    shift = (uint8_t)(24u - ((uint8_t)dest_index * 8u));
                    break;
                case GL_UNSIGNED_INT_8_8_8_8_REV:
                    bits = 8u;
                    shift = (uint8_t)((uint8_t)dest_index * 8u);
                    break;
                case GL_UNSIGNED_INT_10_10_10_2:
                    bits = dest_index == 3 ? 2u : 10u;
                    shift = dest_index == 0 ? 22u : dest_index == 1 ? 12u : dest_index == 2 ? 2u : 0u;
                    break;
                case GL_UNSIGNED_INT_2_10_10_10_REV:
                    bits = dest_index == 3 ? 2u : 10u;
                    shift = dest_index == 0 ? 0u : dest_index == 1 ? 10u : dest_index == 2 ? 20u : 30u;
                    break;
                default:
                    break;
            }
            uint64_t maxv = bits >= 64u ? UINT64_MAX : ((1ull << bits) - 1ull);
            /* For integer formats, clamp the value directly to [0, maxv].
             * For normalized formats, scale from [0,1] to [0, maxv]. */
            uint64_t iv;
            if (integer_format) {
                iv = (uint64_t)mglClampDouble(value, 0.0, (double)maxv);
            } else {
                iv = (uint64_t)(mglClampDouble(value, 0.0, 1.0) * (double)maxv + 0.5);
            }
            uint64_t packed = mglReadUnsignedLE(dst, storage_size);
            packed &= ~(maxv << shift);
            packed |= (iv & maxv) << shift;
            mglWriteUnsignedLE(dst, storage_size, packed);
            return;
        }
        default:
            break;
    }

    size_t component_size = sizeForType(type);
    if (component_size == 0u) {
        return;
    }
    uint8_t *p = dst + ((size_t)dest_index * component_size);

    switch (type) {
        case GL_UNSIGNED_BYTE: {
            uint8_t v = integer_format
                ? (uint8_t)mglClampDouble(value, 0.0, 255.0)
                : (uint8_t)(mglClampDouble(value, 0.0, 1.0) * 255.0 + 0.5);
            memcpy(p, &v, sizeof(v));
            break;
        }
        case GL_BYTE: {
            int8_t v = integer_format
                ? (int8_t)mglClampDouble(value, -128.0, 127.0)
                : (int8_t)(mglClampDouble(value, -1.0, 1.0) * 127.0 + (value >= 0.0 ? 0.5 : -0.5));
            memcpy(p, &v, sizeof(v));
            break;
        }
        case GL_UNSIGNED_SHORT: {
            uint16_t v = integer_format
                ? (uint16_t)mglClampDouble(value, 0.0, 65535.0)
                : (uint16_t)(mglClampDouble(value, 0.0, 1.0) * 65535.0 + 0.5);
            memcpy(p, &v, sizeof(v));
            break;
        }
        case GL_SHORT: {
            int16_t v = integer_format
                ? (int16_t)mglClampDouble(value, -32768.0, 32767.0)
                : (int16_t)(mglClampDouble(value, -1.0, 1.0) * 32767.0 + (value >= 0.0 ? 0.5 : -0.5));
            memcpy(p, &v, sizeof(v));
            break;
        }
        case GL_UNSIGNED_INT: {
            uint32_t v = integer_format
                ? (uint32_t)mglClampDouble(value, 0.0, 4294967295.0)
                : (uint32_t)(mglClampDouble(value, -1.0, 1.0) * 4294967295.0 + (value >= 0.0 ? 0.5 : -0.5));
            memcpy(p, &v, sizeof(v));
            break;
        }
        case GL_INT: {
            int32_t v = integer_format
                ? (int32_t)mglClampDouble(value, -2147483648.0, 2147483647.0)
                : (int32_t)(mglClampDouble(value, -1.0, 1.0) * 2147483647.0 + (value >= 0.0 ? 0.5 : -0.5));
            memcpy(p, &v, sizeof(v));
            break;
        }
        case GL_FLOAT: {
            float v = (float)value;
            memcpy(p, &v, sizeof(v));
            break;
        }
        case GL_HALF_FLOAT: {
            uint16_t v = mglFloatToHalf((float)value);
            memcpy(p, &v, sizeof(v));
            break;
        }
        case GL_UNSIGNED_INT_10F_11F_11F_REV: {
            /* Packed unsigned float: R=11f (bits 0-10), G=11f (bits 11-21),
             * B=10f (bits 22-31). Only valid for GL_RGB format.
             * Do NOT clamp to [0,1] — unsigned float can represent > 1.0. */
            uint32_t packed = mglReadUnsignedLE(dst, sizeof(uint32_t));
            float fv = (float)mglClampDouble(value, 0.0, 65024.0);
            if (dest_index == 0) {
                packed = (packed & ~0x7ffu) | (mglFloatToFloat11(fv) & 0x7ffu);
            } else if (dest_index == 1) {
                packed = (packed & ~(0x7ffu << 11u)) | ((mglFloatToFloat11(fv) & 0x7ffu) << 11u);
            } else if (dest_index == 2) {
                packed = (packed & ~(0x3ffu << 22u)) | ((mglFloatToFloat10(fv) & 0x3ffu) << 22u);
            }
            mglWriteUnsignedLE(dst, sizeof(uint32_t), packed);
            break;
        }
        /* GL_UNSIGNED_INT_5_9_9_9_REV (shared exponent) cannot be handled
         * per-component — the shared exponent requires all 3 RGB values at
         * once.  RGB9_E5 downloads use the dedicated handler in
         * mglCopyTextureRectFromCPU which calls mglUnpackSharedExp + per-
         * component mglWriteExternalComponent with the external type, never
         * reaching this case.  For invalid non-RGB9_E5 downloads with this
         * type, fall through to default (leave pixel zeroed). */
        default:
            break;
    }
}

/* Check if the external format/type produces the exact same bit layout as
 * the internal format's CPU storage.  When true, upload/download can use
 * raw memcpy instead of per-component unpack/repack through double, which
 * avoids precision loss for packed float formats (R11F_G11F_B10F, RGB9_E5)
 * and unnecessary work for packed integer formats (RGB10_A2, RGB10_A2UI). */
bool mglIsIdentityPackedFormat(GLenum internalformat, GLenum format, GLenum type)
{
    /* R11F_G11F_B10F: GL_UNSIGNED_INT_10F_11F_11F_REV has the same
     * LSB-first bit layout as Metal's RG11B10Float CPU storage. */
    if (internalformat == GL_R11F_G11F_B10F &&
        (format == GL_RGB || format == GL_BGR) &&
        type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
        return true;
    }
    /* RGB10_A2 / RGB10_A2UI: GL_UNSIGNED_INT_2_10_10_10_REV has the same
     * LSB-first bit layout as Metal's RGB10A2 CPU storage. */
    if ((internalformat == GL_RGB10_A2 || internalformat == GL_RGB10_A2UI) &&
        (format == GL_RGBA || format == GL_BGRA) &&
        type == GL_UNSIGNED_INT_2_10_10_10_REV) {
        return true;
    }
    /* RGB9_E5: GL_UNSIGNED_INT_5_9_9_9_REV has the same bit layout. */
    if (internalformat == GL_RGB9_E5 &&
        (format == GL_RGB || format == GL_BGR) &&
        type == GL_UNSIGNED_INT_5_9_9_9_REV) {
        return true;
    }
    return false;
}

/* Check if an uncompressed (format, type) pair produces the exact same
 * byte layout as the internal format's CPU storage.  When true, upload/
 * download can use raw memcpy instead of the per-component double loop.
 *
 * This covers the most common Minecraft/LWJGL upload combos:
 *   GL_RGBA8  + GL_RGBA          + GL_UNSIGNED_BYTE
 *   GL_RGB8   + GL_RGB           + GL_UNSIGNED_BYTE
 *   GL_RG8    + GL_RG            + GL_UNSIGNED_BYTE
 *   GL_R8     + GL_RED           + GL_UNSIGNED_BYTE
 *   GL_RGBA16F+ GL_RGBA          + GL_HALF_FLOAT
 *   GL_RGBA32F+ GL_RGBA          + GL_FLOAT
 *   GL_RGBA8UI+ GL_RGBA_INTEGER  + GL_UNSIGNED_BYTE
 *   GL_R32F    + GL_RED          + GL_FLOAT
 *   ... and all similar R/RG/RGB/RGBA variants at 8/16/32 bits.
 *
 * Depth-only formats (GL_DEPTH_COMPONENT16/32/32F) are also covered.
 * Combined depth-stencil and packed (non-byte-aligned) formats are NOT
 * handled here — they use mglIsIdentityPackedFormat or dedicated paths. */
bool mglIsIdentityUncompressedFormat(GLenum internalformat, GLenum format, GLenum type)
{
    GLint canonical = mglTexLevelCanonicalInternalFormat((GLint)internalformat);

    /* Depth-only formats: external format must be GL_DEPTH_COMPONENT. */
    GLint depth_bits = mglTexLevelComponentBits(canonical, GL_TEXTURE_DEPTH_SIZE);
    if (depth_bits > 0) {
        if (format != GL_DEPTH_COMPONENT) return false;
        GLenum depth_type = (GLenum)mglTexLevelComponentType(canonical, GL_TEXTURE_DEPTH_TYPE);
        if (depth_type == GL_FLOAT && depth_bits == 32) {
            return type == GL_FLOAT;
        }
        if (depth_type == GL_UNSIGNED_NORMALIZED) {
            if (depth_bits == 16) return type == GL_UNSIGNED_SHORT;
            if (depth_bits == 32) return type == GL_UNSIGNED_INT;
        }
        return false;
    }

    /* Color formats: count components and verify uniform bit width. */
    GLuint component_count = 0u;
    GLuint component_bits = 0u;
    for (GLuint i = 0u; i < 4u; i++) {
        GLenum pname = (i == 0u) ? GL_TEXTURE_RED_SIZE :
                       (i == 1u) ? GL_TEXTURE_GREEN_SIZE :
                       (i == 2u) ? GL_TEXTURE_BLUE_SIZE : GL_TEXTURE_ALPHA_SIZE;
        GLint bits = mglTexLevelComponentBits(canonical, pname);
        if (bits == 0) continue;
        if (component_bits == 0u) component_bits = (GLuint)bits;
        else if (component_bits != (GLuint)bits) return false;  /* mixed widths */
        component_count++;
    }
    if (component_count == 0u || component_bits == 0u) return false;

    /* Determine expected external format from component count + integer-ness. */
    bool is_integer = mglInternalFormatIsInteger(canonical);
    GLenum expected_format;
    switch (component_count) {
        case 1: expected_format = is_integer ? GL_RED_INTEGER   : GL_RED;   break;
        case 2: expected_format = is_integer ? GL_RG_INTEGER    : GL_RG;    break;
        case 3: expected_format = is_integer ? GL_RGB_INTEGER   : GL_RGB;   break;
        case 4: expected_format = is_integer ? GL_RGBA_INTEGER  : GL_RGBA;  break;
        default: return false;
    }
    if (format != expected_format) return false;

    /* Determine expected external type from component type + bit width. */
    GLint component_type = mglTexLevelComponentType(canonical, GL_TEXTURE_RED_TYPE);
    GLenum expected_type = 0u;
    switch (component_type) {
        case GL_UNSIGNED_NORMALIZED:
            if (component_bits == 8u)  expected_type = GL_UNSIGNED_BYTE;
            else if (component_bits == 16u) expected_type = GL_UNSIGNED_SHORT;
            else if (component_bits == 32u) expected_type = GL_UNSIGNED_INT;
            break;
        case GL_SIGNED_NORMALIZED:
            if (component_bits == 8u)  expected_type = GL_BYTE;
            else if (component_bits == 16u) expected_type = GL_SHORT;
            break;
        case GL_INT:
            if (component_bits == 8u)  expected_type = GL_BYTE;
            else if (component_bits == 16u) expected_type = GL_SHORT;
            else if (component_bits == 32u) expected_type = GL_INT;
            break;
        case GL_UNSIGNED_INT:
            if (component_bits == 8u)  expected_type = GL_UNSIGNED_BYTE;
            else if (component_bits == 16u) expected_type = GL_UNSIGNED_SHORT;
            else if (component_bits == 32u) expected_type = GL_UNSIGNED_INT;
            break;
        case GL_HALF_FLOAT:
            if (component_bits == 16u) expected_type = GL_HALF_FLOAT;
            break;
        case GL_FLOAT:
            if (component_bits == 32u) expected_type = GL_FLOAT;
            break;
        default:
            break;
    }
    if (expected_type == 0u || type != expected_type) return false;

    return true;
}

/* Detects when the only difference between the external (format, type) and the
 * internal format's CPU storage is a B↔R channel swap.  This covers the common
 * GL_BGRA+UBYTE → GL_RGBA8 and GL_BGR+UBYTE → GL_RGB8 cases, which only need a
 * per-pixel byte swap (offset 0 ↔ offset 2) instead of the full per-component
 * double conversion.  Returns true if the swap fast path applies. */
bool mglIsBGRByteSwapFormat(GLenum internalformat, GLenum format, GLenum type)
{
    GLint canonical = mglTexLevelCanonicalInternalFormat((GLint)internalformat);

    /* Depth formats don't have BGR variants. */
    GLint depth_bits = mglTexLevelComponentBits(canonical, GL_TEXTURE_DEPTH_SIZE);
    if (depth_bits > 0) return false;

    /* Count components and verify uniform 8-bit width. */
    GLuint component_count = 0u;
    for (GLuint i = 0u; i < 4u; i++) {
        GLenum pname = (i == 0u) ? GL_TEXTURE_RED_SIZE :
                       (i == 1u) ? GL_TEXTURE_GREEN_SIZE :
                       (i == 2u) ? GL_TEXTURE_BLUE_SIZE : GL_TEXTURE_ALPHA_SIZE;
        GLint bits = mglTexLevelComponentBits(canonical, pname);
        if (bits == 0) continue;
        if (bits != 8) return false;
        component_count++;
    }
    if (component_count != 3u && component_count != 4u) return false;

    /* Determine expected BGR/BGRA counterpart. */
    bool is_integer = mglInternalFormatIsInteger(canonical);
    GLenum expected_bgr_format;
    if (component_count == 3u) {
        expected_bgr_format = is_integer ? GL_BGR_INTEGER : GL_BGR;
    } else {
        expected_bgr_format = is_integer ? GL_BGRA_INTEGER : GL_BGRA;
    }
    if (format != expected_bgr_format) return false;

    /* Verify type matches 8-bit component type. */
    GLint component_type = mglTexLevelComponentType(canonical, GL_TEXTURE_RED_TYPE);
    GLenum expected_type = 0u;
    switch (component_type) {
        case GL_UNSIGNED_NORMALIZED:
        case GL_UNSIGNED_INT:
            expected_type = GL_UNSIGNED_BYTE;
            break;
        case GL_SIGNED_NORMALIZED:
        case GL_INT:
            expected_type = GL_BYTE;
            break;
        default:
            return false;
    }
    if (type != expected_type) return false;

    return true;
}

bool mglClearTexInternalFormatIsColor(GLenum internalformat)
{
    switch (mglTexLevelCanonicalInternalFormat((GLint)internalformat)) {
        case GL_R8: case GL_R8_SNORM: case GL_R16: case GL_R16_SNORM:
        case GL_R16F: case GL_R32F:
        case GL_R8I: case GL_R16I: case GL_R32I:
        case GL_R8UI: case GL_R16UI: case GL_R32UI:
        case GL_RG8: case GL_RG8_SNORM: case GL_RG16: case GL_RG16_SNORM:
        case GL_RG16F: case GL_RG32F:
        case GL_RG8I: case GL_RG16I: case GL_RG32I:
        case GL_RG8UI: case GL_RG16UI: case GL_RG32UI:
        case GL_R3_G3_B2:
        case GL_RGB4: case GL_RGB5: case GL_RGB8: case GL_RGB8_SNORM:
        case GL_RGB10: case GL_RGB12: case GL_RGB16: case GL_RGB16_SNORM:
        case GL_SRGB8: case GL_RGB565:
        case GL_RGB16F: case GL_RGB32F:
        case GL_R11F_G11F_B10F: case GL_RGB9_E5:
        case GL_RGB8I: case GL_RGB16I: case GL_RGB32I:
        case GL_RGB8UI: case GL_RGB16UI: case GL_RGB32UI:
        case GL_RGBA2: case GL_RGBA4: case GL_RGB5_A1:
        case GL_RGBA8: case GL_RGBA8_SNORM:
        case GL_RGB10_A2: case GL_RGBA12: case GL_RGBA16:
        case GL_RGBA16_SNORM: case GL_SRGB8_ALPHA8:
        case GL_RGBA16F: case GL_RGBA32F:
        case GL_RGBA8I: case GL_RGBA16I: case GL_RGBA32I:
        case GL_RGBA8UI: case GL_RGBA16UI: case GL_RGBA32UI:
        case GL_RGB10_A2UI:
            return true;
        default:
            return false;
    }
}

GLenum mglClearTexFormatCompatibilityError(GLenum internalformat, GLenum format)
{
    GLint canonical = mglTexLevelCanonicalInternalFormat((GLint)internalformat);
    bool has_color = mglClearTexInternalFormatIsColor((GLenum)canonical);
    bool has_depth_only = (canonical == GL_DEPTH_COMPONENT16 ||
                           canonical == GL_DEPTH_COMPONENT24 ||
                           canonical == GL_DEPTH_COMPONENT32 ||
                           canonical == GL_DEPTH_COMPONENT32F);
    bool has_depth_stencil = (canonical == GL_DEPTH24_STENCIL8 ||
                              canonical == GL_DEPTH32F_STENCIL8);
    bool has_stencil_only = (canonical == GL_STENCIL_INDEX8 ||
                             canonical == GL_STENCIL_INDEX ||
                             canonical == GL_STENCIL_INDEX1 ||
                             canonical == GL_STENCIL_INDEX4 ||
                             canonical == GL_STENCIL_INDEX16);
    bool internal_integer = has_color && mglInternalFormatIsInteger(canonical);
    bool format_integer = mglExternalFormatIsInteger(format);

    if (has_depth_only) {
        if (format != GL_DEPTH_COMPONENT) {
            return GL_INVALID_OPERATION;
        }
        return GL_NO_ERROR;
    }
    if (has_depth_stencil) {
        if (format != GL_DEPTH_STENCIL) {
            return GL_INVALID_OPERATION;
        }
        return GL_NO_ERROR;
    }
    if (has_stencil_only) {
        if (format != GL_STENCIL_INDEX) {
            return GL_INVALID_OPERATION;
        }
        return GL_NO_ERROR;
    }

    if (has_color) {
        if (format == GL_DEPTH_COMPONENT || format == GL_DEPTH_STENCIL || format == GL_STENCIL_INDEX) {
            return GL_INVALID_OPERATION;
        }
        if (internal_integer != format_integer) {
            return GL_INVALID_OPERATION;
        }
    }

    return GL_NO_ERROR;
}

size_t mglClearComponentSize(GLenum type)
{
    switch (type) {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
            return 1u;
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_HALF_FLOAT:
            return 2u;
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_FLOAT:
            return 4u;
        default:
            return 0u;
    }
}

void mglStoreDefaultAlpha(uint8_t *pixel, size_t storage_pixel_size, GLenum type)
{
    size_t component_size = mglClearComponentSize(type);
    if (!pixel || component_size == 0u || storage_pixel_size < component_size * 4u) {
        return;
    }

    uint8_t *alpha = pixel + component_size * 3u;
    switch (type) {
        case GL_UNSIGNED_BYTE: {
            uint8_t v = UINT8_MAX;
            memcpy(alpha, &v, sizeof(v));
            break;
        }
        case GL_BYTE: {
            int8_t v = INT8_MAX;
            memcpy(alpha, &v, sizeof(v));
            break;
        }
        case GL_UNSIGNED_SHORT:
        case GL_HALF_FLOAT: {
            uint16_t v = UINT16_MAX;
            memcpy(alpha, &v, sizeof(v));
            break;
        }
        case GL_SHORT: {
            int16_t v = INT16_MAX;
            memcpy(alpha, &v, sizeof(v));
            break;
        }
        case GL_UNSIGNED_INT: {
            uint32_t v = UINT32_MAX;
            memcpy(alpha, &v, sizeof(v));
            break;
        }
        case GL_INT: {
            int32_t v = INT32_MAX;
            memcpy(alpha, &v, sizeof(v));
            break;
        }
        case GL_FLOAT: {
            float v = 1.0f;
            memcpy(alpha, &v, sizeof(v));
            break;
        }
        default:
            break;
    }
}

bool mglTextureFormatLooksDepthOrStencil(GLenum internalformat)
{
    switch (internalformat) {
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
        case GL_DEPTH_STENCIL:
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
        case GL_STENCIL_INDEX:
        case GL_STENCIL_INDEX8:
            return true;
        default:
            return false;
    }
}

bool mglTexStorageInternalFormatValid(GLenum internalformat)
{
    return mglTexLevelInternalFormatCompressed(internalformat) ||
           mtlFormatForGLInternalFormat(internalformat) != MTLPixelFormatInvalid;
}

/*
 * Returns the byte stride of one row of compressed blocks for the given
 * compressed internalformat and image width. Returns 0 for non-compressed or
 * unknown formats (caller should fall back to pitch=0).
 *
 * For block-compressed textures, Metal requires bytesPerRow to be the number
 * of bytes from the start of one row of blocks to the next, i.e.
 *   ceil(width / block_w) * bytes_per_block
 * The CPU-side data layout stored by mglStoreCompressedTextureImage matches.
 */
GLuint mglCompressedBytesPerRowOf(GLenum internalformat, GLsizei width)
{
    GLuint bw = 0, bsz = 0;
    switch (internalformat) {
        /* S3TC/DXT */
        case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
        case 0x8c4c: /* GL_COMPRESSED_SRGB_S3TC_DXT1_EXT */
        case 0x8c4d: /* GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT */
            bw = 4;  bsz = 8;  break;
        case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:
        case 0x8c4e: /* GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT */
        case 0x8c4f: /* GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT */
            bw = 4;  bsz = 16; break;
        /* ASTC LDR: all 16 bytes/block, block size varies */
        case GL_COMPRESSED_RGBA_ASTC_4x4_KHR:       bw = 4;  bsz = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_5x4_KHR:       bw = 5;  bsz = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_5x5_KHR:       bw = 5;  bsz = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_6x5_KHR:       bw = 6;  bsz = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_6x6_KHR:       bw = 6;  bsz = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_8x5_KHR:       bw = 8;  bsz = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_8x6_KHR:       bw = 8;  bsz = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_8x8_KHR:       bw = 8;  bsz = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_10x5_KHR:      bw = 10; bsz = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_10x6_KHR:      bw = 10; bsz = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_10x8_KHR:      bw = 10; bsz = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_10x10_KHR:     bw = 10; bsz = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_12x10_KHR:     bw = 12; bsz = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_12x12_KHR:     bw = 12; bsz = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:   bw = 4;  bsz = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:   bw = 5;  bsz = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:   bw = 5;  bsz = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:   bw = 6;  bsz = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:   bw = 6;  bsz = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:   bw = 8;  bsz = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:   bw = 8;  bsz = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:   bw = 8;  bsz = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:  bw = 10; bsz = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:  bw = 10; bsz = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:  bw = 10; bsz = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR: bw = 10; bsz = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR: bw = 12; bsz = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR: bw = 12; bsz = 16; break;
        default:
            return 0;
    }
    if (width <= 0 || bw == 0) return 0;
    /* Rounded-up block count per row x bytes per block. */
    return ((GLuint)((width + bw - 1) / bw)) * bsz;
}

bool mglCompressedBlockInfoOf(GLenum internalformat,
                                     GLuint *out_bw,
                                     GLuint *out_bh,
                                     GLuint *out_bd,
                                     GLuint *out_bs)
{
    GLuint bw = 0, bh = 0, bd = 1, bs = 0;
    switch (internalformat) {
        case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
        case 0x8c4c: /* GL_COMPRESSED_SRGB_S3TC_DXT1_EXT */
        case 0x8c4d: /* GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT */
            bw = 4;  bh = 4;  bs = 8;  break;
        case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:
        case 0x8c4e: /* GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT */
        case 0x8c4f: /* GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT */
            bw = 4;  bh = 4;  bs = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_4x4_KHR:       bw = 4;  bh = 4;  bs = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_5x4_KHR:       bw = 5;  bh = 4;  bs = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_5x5_KHR:       bw = 5;  bh = 5;  bs = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_6x5_KHR:       bw = 6;  bh = 5;  bs = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_6x6_KHR:       bw = 6;  bh = 6;  bs = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_8x5_KHR:       bw = 8;  bh = 5;  bs = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_8x6_KHR:       bw = 8;  bh = 6;  bs = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_8x8_KHR:       bw = 8;  bh = 8;  bs = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_10x5_KHR:      bw = 10; bh = 5;  bs = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_10x6_KHR:      bw = 10; bh = 6;  bs = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_10x8_KHR:      bw = 10; bh = 8;  bs = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_10x10_KHR:     bw = 10; bh = 10; bs = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_12x10_KHR:     bw = 12; bh = 10; bs = 16; break;
        case GL_COMPRESSED_RGBA_ASTC_12x12_KHR:     bw = 12; bh = 12; bs = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:   bw = 4;  bh = 4;  bs = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:   bw = 5;  bh = 4;  bs = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:   bw = 5;  bh = 5;  bs = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:   bw = 6;  bh = 5;  bs = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:   bw = 6;  bh = 6;  bs = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:   bw = 8;  bh = 5;  bs = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:   bw = 8;  bh = 6;  bs = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:   bw = 8;  bh = 8;  bs = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:  bw = 10; bh = 5;  bs = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:  bw = 10; bh = 6;  bs = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:  bw = 10; bh = 8;  bs = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR: bw = 10; bh = 10; bs = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR: bw = 12; bh = 10; bs = 16; break;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR: bw = 12; bh = 12; bs = 16; break;
        default:
            return false;
    }
    if (out_bw) *out_bw = bw;
    if (out_bh) *out_bh = bh;
    if (out_bd) *out_bd = bd;
    if (out_bs) *out_bs = bs;
    return true;
}

/*
 * The six generic compressed internal formats are never valid as the `format`
 * argument to glCompressedTex(Sub)Image* — only concrete (sized) block formats
 * are.  CompressedTextureSubImage* must raise GL_INVALID_ENUM for them.
 */
bool mglIsGenericCompressedFormat(GLenum format)
{
    switch (format) {
        case GL_COMPRESSED_RED:
        case GL_COMPRESSED_RG:
        case GL_COMPRESSED_RGB:
        case GL_COMPRESSED_RGBA:
        case GL_COMPRESSED_SRGB:
        case GL_COMPRESSED_SRGB_ALPHA:
            return true;
        default:
            return false;
    }
}

/*
 * Compressed block formats that are defined over a 2-D block (RGTC2, BPTC,
 * S3TC/DXT, ASTC LDR, ETC2, EAC) require a height and are therefore illegal
 * as the format of a CompressedTexSubImage1D update.  Used to raise
 * GL_INVALID_ENUM for the 1D target, matching the spec rule that these
 * formats are not 1D.
 */
bool mglCompressedFormatRequiresHeight(GLenum format)
{
    switch (mglTexLevelCanonicalInternalFormat((GLint)format)) {
        case GL_COMPRESSED_RG_RGTC2:
        case GL_COMPRESSED_SIGNED_RG_RGTC2:
        case GL_COMPRESSED_RGBA_BPTC_UNORM:
        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
        case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
        case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:
        case 0x8c4c: /* GL_COMPRESSED_SRGB_S3TC_DXT1_EXT */
        case 0x8c4d: /* GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT */
        case 0x8c4e: /* GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT */
        case 0x8c4f: /* GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT */
        case GL_COMPRESSED_RGBA_ASTC_4x4_KHR:
        case GL_COMPRESSED_RGBA_ASTC_5x4_KHR:
        case GL_COMPRESSED_RGBA_ASTC_5x5_KHR:
        case GL_COMPRESSED_RGBA_ASTC_6x5_KHR:
        case GL_COMPRESSED_RGBA_ASTC_6x6_KHR:
        case GL_COMPRESSED_RGBA_ASTC_8x5_KHR:
        case GL_COMPRESSED_RGBA_ASTC_8x6_KHR:
        case GL_COMPRESSED_RGBA_ASTC_8x8_KHR:
        case GL_COMPRESSED_RGBA_ASTC_10x5_KHR:
        case GL_COMPRESSED_RGBA_ASTC_10x6_KHR:
        case GL_COMPRESSED_RGBA_ASTC_10x8_KHR:
        case GL_COMPRESSED_RGBA_ASTC_10x10_KHR:
        case GL_COMPRESSED_RGBA_ASTC_12x10_KHR:
        case GL_COMPRESSED_RGBA_ASTC_12x12_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:
        case GL_COMPRESSED_RGB8_ETC2:
        case GL_COMPRESSED_SRGB8_ETC2:
        case GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2:
        case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2:
        case GL_COMPRESSED_RGBA8_ETC2_EAC:
        case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:
        case GL_COMPRESSED_R11_EAC:
        case GL_COMPRESSED_SIGNED_R11_EAC:
        case GL_COMPRESSED_RG11_EAC:
        case GL_COMPRESSED_SIGNED_RG11_EAC:
            return true;
        default:
            return false;
    }
}

#pragma mark copy tex
bool mglCopyTex2DFaceForTarget(GLenum target, GLuint *face_out)
{
    if (face_out) {
        *face_out = 0u;
    }

    switch (target) {
        case GL_TEXTURE_2D:
        case GL_TEXTURE_RECTANGLE:
            return true;

        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            if (face_out) {
                *face_out = (GLuint)(target - GL_TEXTURE_CUBE_MAP_POSITIVE_X);
            }
            return true;

        default:
            return false;
    }
}
