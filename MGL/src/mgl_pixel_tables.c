/* SPDX-License-Identifier: LGPL-3.0-only */
/*
 * mgl_pixel_tables.c
 * MGL
 *
 * GL pixel-transfer tables: how many components a format carries and how many
 * bytes an element of a type occupies.  Pure GL semantics with no renderer or
 * backend dependency, so the Metal-cpp smoke gate can link this unit alone
 * (mgl_pixel_format.c drags the textures.c symbol set with it).
 */

#include "mgl_pixel_format.h"
#include "mgl_render_api_texture.h"
#include "mgl_render_api_readback.h"

#include <stdio.h>

uint32_t mglSizeForType(uint32_t type) {
    switch (type) {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
            return sizeof(uint8_t);
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_HALF_FLOAT:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
            return sizeof(uint16_t);
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_FLOAT:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
        case GL_UNSIGNED_INT_24_8:
            return sizeof(uint32_t);
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
            return 8u;
        default:
            return sizeof(uint32_t);
    }
}

uint32_t mglNumComponentsForFormat(uint32_t format) {
    switch (format) {
        case GL_RED:
        case GL_RED_INTEGER:
        case GL_GREEN:
        case GL_BLUE:
        case GL_STENCIL_INDEX:
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_STENCIL:
        case GL_ALPHA:
        case 0x803C: /* GL_ALPHA8 */
        case 0x803E: /* GL_ALPHA16 */
        case 0x8816: /* GL_ALPHA32F_ARB */
        case 0x881C: /* GL_ALPHA16F_ARB */
        case 0x1909: /* GL_LUMINANCE */
        case 0x8040: /* GL_LUMINANCE8 */
        case 0x8048: /* GL_LUMINANCE16 (pixel_utils local define) */
        case 0x8818: /* GL_LUMINANCE32F_ARB */
        case 0x881E: /* GL_LUMINANCE16F_ARB */
        case GL_R8:
        case GL_R8_SNORM:
        case GL_R16:
        case GL_R16_SNORM:
        case GL_R16F:
        case GL_R32F:
        case GL_R8I:
        case GL_R8UI:
        case GL_R16I:
        case GL_R16UI:
        case GL_R32I:
        case GL_R32UI:
        case GL_SR8_EXT:
        case 0x8D7E: /* GL_ALPHA8UI_EXT */
        case 0x9014: /* GL_ALPHA8_SNORM */
        case 0x9018: /* GL_ALPHA16_SNORM */
            return 1u;

        case GL_RG:
        case GL_RG_INTEGER:
        case 0x190A: /* GL_LUMINANCE_ALPHA */
        case 0x8819: /* GL_LUMINANCE_ALPHA32F_ARB */
        case 0x881F: /* GL_LUMINANCE_ALPHA16F_ARB */
        case 0x9016: /* GL_LUMINANCE8_ALPHA8_SNORM */
        case 0x901a: /* GL_LUMINANCE16_ALPHA16_SNORM */
        case GL_RG8:
        case GL_RG8_SNORM:
        case GL_RG16:
        case GL_RG16_SNORM:
        case GL_RG16F:
        case GL_RG32F:
        case GL_RG8I:
        case GL_RG8UI:
        case GL_RG16I:
        case GL_RG16UI:
        case GL_RG32I:
        case GL_RG32UI:
        case GL_SRG8_EXT:
            return 2u;

        case 0x8d7b: /* GL_ALPHA8I_EXT */
        case 0x8d81: /* GL_ALPHA32I_EXT */
        case 0x8d87: /* GL_ALPHA16I_EXT */
        case 0x8d8d: /* GL_ALPHA32UI_EXT */
        case 0x8d93: /* GL_ALPHA16UI_EXT */
        case 0x8d72: /* GL_ALPHA32UI_EXT */
            return 1u;

        case GL_RGB:
        case GL_BGR:
        case GL_RGB_INTEGER:
        case GL_BGR_INTEGER:
        case GL_RGB8:
        case GL_RGB8_SNORM:
        case GL_SRGB8:
        case GL_RGB16F:
        case GL_RGB32F:
        case GL_R11F_G11F_B10F:
        case GL_RGB9_E5:
        case GL_RGB8I:
        case GL_RGB8UI:
        case GL_RGB16I:
        case GL_RGB16UI:
        case GL_RGB32I:
        case GL_RGB32UI:
        case GL_RGB565:
            return 3u;

        case 0x8d75: /* alternate GL_RGB8I */
        case 0x8d7a: /* alternate GL_RGB8UI */
        case 0x8d80: /* alternate GL_RGB32UI */
        case 0x8d86: /* alternate GL_RGB16I */
        case 0x8d8c: /* alternate GL_RGB32I */
        case 0x8d92: /* alternate GL_RGB16UI */
            return 3u;

        case GL_RGBA:
        case GL_BGRA:
        case GL_RGBA_INTEGER:
        case GL_BGRA_INTEGER:
        case GL_RGBA8:
        case GL_RGBA8_SNORM:
        case GL_SRGB8_ALPHA8:
        case GL_RGBA16F:
        case GL_RGBA32F:
        case GL_RGBA8I:
        case GL_RGBA8UI:
        case GL_RGBA16I:
        case GL_RGBA16UI:
        case GL_RGBA32I:
        case GL_RGBA32UI:
        case GL_RGB10_A2:
        case GL_RGB10_A2UI:
        case GL_RGB5_A1:
        case GL_RGBA4:
            return 4u;

        case 0x8d78: /* alternate GL_RGBA8UI */
        case 0x8d84: /* alternate GL_RGBA16I */
        case 0x8d8a: /* alternate GL_RGBA32I */
        case 0x8d90: /* alternate GL_RGBA16UI */
            return 4u;

        case 0x8d95: /* GL_GREEN_INTEGER */
        case 0x8d96: /* GL_BLUE_INTEGER */
            return 1u;

        default:
            fprintf(stderr,
                    "MGL WARNING: numComponentsForFormat unknown format 0x%x, "
                    "assuming 4 components\n",
                    format);
            return 4u;
    }
}

int mglReadbackFormatChannelMap(uint32_t format, int* slots,
                                          int src_idx[4]) {
    if (!slots || !src_idx) return 0;
    src_idx[0] = src_idx[1] = src_idx[2] = src_idx[3] = 0;
    switch (format) {
        case GL_RGBA: *slots = 4; src_idx[0]=0; src_idx[1]=1; src_idx[2]=2; src_idx[3]=3; return 1;
        case GL_BGRA: *slots = 4; src_idx[0]=2; src_idx[1]=1; src_idx[2]=0; src_idx[3]=3; return 1;
        case GL_RGB:  *slots = 3; src_idx[0]=0; src_idx[1]=1; src_idx[2]=2; return 1;
        case GL_BGR:  *slots = 3; src_idx[0]=2; src_idx[1]=1; src_idx[2]=0; return 1;
        case GL_RG:   *slots = 2; src_idx[0]=0; src_idx[1]=1; return 1;
        case GL_RED:  *slots = 1; src_idx[0]=0; return 1;
        case GL_GREEN: *slots = 1; src_idx[0]=1; return 1;
        case GL_BLUE:  *slots = 1; src_idx[0]=2; return 1;
        case GL_ALPHA: *slots = 1; src_idx[0]=3; return 1;
        default: return 0;
    }
}

int mglPixelTypeIsPacked(uint32_t type) {
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
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
        case GL_UNSIGNED_INT_24_8:
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
            return 1;
        default:
            return 0;
    }
}

int mglReadbackRGB10A2TypeAccepted(uint32_t type) {
    switch (type) {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_FLOAT:
        case GL_HALF_FLOAT:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
            return 1;
        default:
            return 0;
    }
}

int mglReadbackRG11B10TypeAccepted(uint32_t type) {
    switch (type) {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_FLOAT:
        case GL_HALF_FLOAT:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
            return 1;
        default:
            return 0;
    }
}

int mglReadback16or32TypeAccepted(uint32_t type) {
    switch (type) {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_FLOAT:
        case GL_HALF_FLOAT:
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
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
            return 1;
        default:
            return 0;
    }
}

int mglReadbackUnorm8ScalarTypeAccepted(uint32_t type) {
    switch (type) {
        case GL_BYTE:
        case GL_SHORT:
        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_UNSIGNED_SHORT:
        case GL_HALF_FLOAT:
        case GL_FLOAT:
            return 1;
        default:
            return 0;
    }
}

int mglReadbackUnorm8PackedTypeAccepted(uint32_t type) {
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
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
            return 1;
        default:
            return 0;
    }
}

uint32_t mglRenderStoredColorComponents(uint32_t internal_format) {
    uint32_t components = mglNumComponentsForFormat(internal_format);
    return components > 0u ? components : 4u;
}

int mglRenderIntegerFormatLayout(
    uint32_t internal_format, uint32_t* out_components,
    uint32_t* out_component_bytes, int* out_signed) {
    if (!out_components || !out_component_bytes || !out_signed) {
        return -1;
    }
    *out_components = 0;
    *out_component_bytes = 0;
    *out_signed = 0;
    switch (internal_format) {
        case GL_R8I:
        case GL_RG8I:
        case GL_RGB8I:
        case GL_RGBA8I:
            *out_components = mglRenderStoredColorComponents(internal_format);
            *out_component_bytes = 1u;
            *out_signed = 1;
            return 0;
        case GL_R8UI:
        case GL_RG8UI:
        case GL_RGB8UI:
        case GL_RGBA8UI:
            *out_components = mglRenderStoredColorComponents(internal_format);
            *out_component_bytes = 1u;
            *out_signed = 0;
            return 0;
        case GL_R16I:
        case GL_RG16I:
        case GL_RGB16I:
        case GL_RGBA16I:
            *out_components = mglRenderStoredColorComponents(internal_format);
            *out_component_bytes = 2u;
            *out_signed = 1;
            return 0;
        case GL_R16UI:
        case GL_RG16UI:
        case GL_RGB16UI:
        case GL_RGBA16UI:
            *out_components = mglRenderStoredColorComponents(internal_format);
            *out_component_bytes = 2u;
            *out_signed = 0;
            return 0;
        case GL_R32I:
        case GL_RG32I:
        case GL_RGB32I:
        case GL_RGBA32I:
            *out_components = mglRenderStoredColorComponents(internal_format);
            *out_component_bytes = 4u;
            *out_signed = 1;
            return 0;
        case GL_R32UI:
        case GL_RG32UI:
        case GL_RGB32UI:
        case GL_RGBA32UI:
            *out_components = mglRenderStoredColorComponents(internal_format);
            *out_component_bytes = 4u;
            *out_signed = 0;
            return 0;
        default:
            return -1;
    }
}

int mglRenderTextureUploadNeedsSingleChannelSwizzle(uint32_t internal_format,
                                                       int swizzled) {
    if (!swizzled) {
        return 0;
    }
    switch (internal_format) {
        case GL_R8:
        case GL_R8_SNORM:
        case GL_R16:
        case GL_R16_SNORM:
        case GL_R16F:
        case GL_R32F:
        case GL_R8I:
        case GL_R8UI:
        case GL_R16I:
        case GL_R16UI:
        case GL_R32I:
        case GL_R32UI:
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
            return 1;
        default:
            return 0;
    }
}

int mglRenderTextureUploadNeedsSingleChannelSwizzleBake(
    uint32_t internal_format, int swizzled) {
    if (!swizzled) {
        return 0;
    }
    switch (internal_format) {
        case GL_R8:
        case GL_R8_SNORM:
        case GL_R16:
        case GL_R16_SNORM:
        case GL_R16F:
        case GL_R32F:
        case GL_R16UI:
        case GL_R32UI:
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
            return 1;
        default:
            return 0;
    }
}

int mglRenderReadbackGLTypeAccepted(uint32_t type) {
    switch (type) {
        case GL_UNSIGNED_BYTE:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_FLOAT:
        case GL_BYTE:
        case GL_SHORT:
        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_UNSIGNED_SHORT:
        case GL_HALF_FLOAT:
        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
            return 1;
        default:
            return 0;
    }
}

int mglRenderReadbackTypeIsCore(uint32_t type) {
    return type == GL_UNSIGNED_BYTE || type == GL_BYTE ||
                   type == GL_UNSIGNED_SHORT || type == GL_SHORT ||
                   type == GL_UNSIGNED_INT || type == GL_INT ||
                   type == GL_FLOAT || type == GL_HALF_FLOAT
               ? 1
               : 0;
}

int mglRenderReadbackTypeAllowsRGB10A2(uint32_t type) {
    return mglRenderReadbackTypeIsCore(type) ||
                   type == GL_UNSIGNED_INT_10_10_10_2 ||
                   type == GL_UNSIGNED_INT_2_10_10_10_REV ||
                   type == GL_UNSIGNED_INT_5_9_9_9_REV ||
                   type == GL_UNSIGNED_INT_8_8_8_8 ||
                   type == GL_UNSIGNED_INT_8_8_8_8_REV
               ? 1
               : 0;
}

int mglRenderReadbackTypeAllowsRG11B10(uint32_t type) {
    return mglRenderReadbackTypeIsCore(type) ||
                   type == GL_UNSIGNED_INT_10F_11F_11F_REV ||
                   type == GL_UNSIGNED_INT_5_9_9_9_REV ||
                   type == GL_UNSIGNED_INT_8_8_8_8 ||
                   type == GL_UNSIGNED_INT_8_8_8_8_REV
               ? 1
               : 0;
}

int mglRenderReadbackTypeAllows16or32(uint32_t type) {
    return mglRenderReadbackTypeIsCore(type) ||
                   type == GL_UNSIGNED_BYTE_3_3_2 ||
                   type == GL_UNSIGNED_BYTE_2_3_3_REV ||
                   type == GL_UNSIGNED_SHORT_5_6_5 ||
                   type == GL_UNSIGNED_SHORT_5_6_5_REV ||
                   type == GL_UNSIGNED_SHORT_4_4_4_4 ||
                   type == GL_UNSIGNED_SHORT_4_4_4_4_REV ||
                   type == GL_UNSIGNED_SHORT_5_5_5_1 ||
                   type == GL_UNSIGNED_SHORT_1_5_5_5_REV ||
                   type == GL_UNSIGNED_INT_8_8_8_8 ||
                   type == GL_UNSIGNED_INT_8_8_8_8_REV ||
                   type == GL_UNSIGNED_INT_10_10_10_2 ||
                   type == GL_UNSIGNED_INT_2_10_10_10_REV ||
                   type == GL_UNSIGNED_INT_10F_11F_11F_REV ||
                   type == GL_UNSIGNED_INT_5_9_9_9_REV
               ? 1
               : 0;
}

int mglRenderReadbackTypeIsWideScalar(uint32_t type) {
    return type == GL_BYTE || type == GL_SHORT || type == GL_INT ||
                   type == GL_UNSIGNED_INT || type == GL_UNSIGNED_SHORT ||
                   type == GL_HALF_FLOAT || type == GL_FLOAT
               ? 1
               : 0;
}

int mglRenderReadbackTypeIsPacked(uint32_t type) {
    return mglRenderReadbackTypeAllows16or32(type) &&
                   !mglRenderReadbackTypeIsCore(type)
               ? 1
               : 0;
}
