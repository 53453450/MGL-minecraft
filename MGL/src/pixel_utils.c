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
 * pixel_utils.c
 * MGL
 *
 */
 
#include <Availability.h>

#include "pixel_utils.h"
#include "glm_context.h"

// Legacy format defines not in core profile headers
#ifndef GL_ALPHA
#define GL_ALPHA                          0x1906
#endif
#ifndef GL_LUMINANCE
#define GL_LUMINANCE                      0x1909
#endif

#ifndef GL_ALPHA8UI_EXT
#define GL_ALPHA8UI_EXT                   0x8D7E
#endif
#ifndef GL_LUMINANCE_ALPHA
#define GL_LUMINANCE_ALPHA                0x190A
#endif
#ifndef GL_ALPHA8
#define GL_ALPHA8                         0x803C
#endif
#ifndef GL_ALPHA16
#define GL_ALPHA16                        0x803E
#endif
#ifndef GL_LUMINANCE8
#define GL_LUMINANCE8                     0x8040
#endif
#ifndef GL_LUMINANCE16
#define GL_LUMINANCE16                    0x8048
#endif
#ifndef GL_ALPHA32F_ARB
#define GL_ALPHA32F_ARB                   0x8816
#endif
#ifndef GL_LUMINANCE32F_ARB
#define GL_LUMINANCE32F_ARB               0x8818
#endif
#ifndef GL_LUMINANCE_ALPHA32F_ARB
#define GL_LUMINANCE_ALPHA32F_ARB         0x8819
#endif
#ifndef GL_ALPHA16F_ARB
#define GL_ALPHA16F_ARB                   0x881C
#endif
#ifndef GL_LUMINANCE16F_ARB
#define GL_LUMINANCE16F_ARB               0x881E
#endif
#ifndef GL_LUMINANCE_ALPHA16F_ARB
#define GL_LUMINANCE_ALPHA16F_ARB         0x881F
#endif
#ifndef GL_SR8_EXT
#define GL_SR8_EXT                        0x8FBD
#endif
#ifndef GL_SRG8_EXT
#define GL_SRG8_EXT                       0x8FBE
#endif

GLuint numComponentsForFormat(GLenum format)
{
    switch(format)
    {
        case GL_RED:
        case GL_RED_INTEGER:
        case GL_GREEN:
        case GL_BLUE:
        case GL_STENCIL_INDEX:
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_STENCIL:
        // Legacy single-channel formats
        case GL_ALPHA:
        case GL_ALPHA8:
        case GL_ALPHA16:
        case GL_ALPHA32F_ARB:
        case GL_ALPHA16F_ARB:
        case GL_LUMINANCE:
        case GL_LUMINANCE8:
        case GL_LUMINANCE16:
        case GL_LUMINANCE32F_ARB:
        case GL_LUMINANCE16F_ARB:
        // Sized R formats (internal formats sometimes passed as format)
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
        case GL_ALPHA8UI_EXT:
        case 0x9014: // GL_ALPHA8_SNORM
        case 0x9018: // GL_ALPHA16_SNORM
            return 1;

        case GL_RG:
        case GL_RG_INTEGER:
        // Legacy two-channel formats
        case GL_LUMINANCE_ALPHA:
        case GL_LUMINANCE_ALPHA32F_ARB:
        case GL_LUMINANCE_ALPHA16F_ARB:
        case 0x9016: // GL_LUMINANCE8_ALPHA8_SNORM
        case 0x901a: // GL_LUMINANCE16_ALPHA16_SNORM
        // Sized RG formats
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
            return 2;

        case 0x8d7b: // GL_ALPHA8I_EXT
        case 0x8d81: // GL_ALPHA32I_EXT
        case 0x8d87: // GL_ALPHA16I_EXT
        case 0x8d8d: // GL_ALPHA32UI_EXT
        case 0x8d93: // GL_ALPHA16UI_EXT
        case 0x8d72: // GL_ALPHA32UI_EXT
            return 1;

        case GL_RGB:
        case GL_BGR:
        case GL_RGB_INTEGER:
        case GL_BGR_INTEGER:
        // Sized RGB formats
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
            return 3;

        case 0x8d75: // alternate GL_RGB8I
        case 0x8d7a: // alternate GL_RGB8UI
        case 0x8d80: // alternate GL_RGB32UI
        case 0x8d86: // alternate GL_RGB16I
        case 0x8d8c: // alternate GL_RGB32I
        case 0x8d92: // alternate GL_RGB16UI
            return 3;

        case GL_RGBA:
        case GL_BGRA:
        case GL_RGBA_INTEGER:
        case GL_BGRA_INTEGER:
        // Sized RGBA formats
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
            return 4;

        case 0x8d78: // alternate GL_RGBA8UI
        // case 0x8d7e: // alternate GL_RGBA32UI - Duplicate of GL_ALPHA8UI_EXT (1 component)
        case 0x8d84: // alternate GL_RGBA16I
        case 0x8d8a: // alternate GL_RGBA32I
        case 0x8d90: // alternate GL_RGBA16UI
            return 4;

        case 0x8d95: // GL_GREEN_INTEGER
        case 0x8d96: // GL_BLUE_INTEGER
            return 1;

        default:
            // Unknown format - return 4 as safe fallback instead of crashing
            fprintf(stderr, "MGL WARNING: numComponentsForFormat unknown format 0x%x, assuming 4 components\n", format);
            return 4;
    }

    return 0;
}

GLuint sizeForType(GLenum type)
{
    switch(type)
    {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
            return sizeof(uint8_t);

        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
            return sizeof(uint16_t);

        case GL_UNSIGNED_INT:
        case GL_INT:
            return sizeof(uint32_t);

        case GL_FLOAT:
            return sizeof(float);

        case GL_HALF_FLOAT:
            return sizeof(uint16_t);

        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
            return sizeof(uint8_t);

        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
            return sizeof(uint16_t);

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
            fprintf(stderr, "MGL WARNING: sizeForType unknown type 0x%x, assuming 4 bytes\n", type);
            return sizeof(uint32_t);
    }

    return 0;
}

GLuint sizeForFormatType(GLenum format, GLenum type)
{
    // Handle type=0 case (used for sized internal formats)
    // The format parameter is actually the internal format in this case
    if (type == 0) {
        switch (format) {
            // Alpha formats (1 component)
            case 0x803c: // GL_ALPHA8
            case 0x8040: // GL_LUMINANCE8
                return 1;
            case 0x803e: // GL_ALPHA16
            case 0x8042: // GL_LUMINANCE16
            case 0x8816: // GL_ALPHA16F_ARB
            case 0x8818: // GL_LUMINANCE16F_ARB
                return 2;
            case 0x881c: // GL_ALPHA32F_ARB
            case 0x881e: // GL_LUMINANCE32F_ARB
                return 4;
            // Luminance-alpha formats (2 components)
            case 0x8045: // GL_LUMINANCE8_ALPHA8
                return 2;
            case 0x8048: // GL_LUMINANCE16_ALPHA16
            case 0x8819: // GL_LUMINANCE_ALPHA16F_ARB
                return 4;
            case 0x881f: // GL_LUMINANCE_ALPHA32F_ARB
                return 8;
            // RGB10_A2UI and SNORM formats
            case 0x8fbd: // GL_RGB10_A2UI
                return 4;
            case 0x8fbe: // GL_RGBA16_SNORM
                return 8;
            // Integer formats
            case 0x8d72: case 0x8d78: // RGBA8I/UI variants
                return 4;
            case 0x8d75: case 0x8d7a: // RGB8I/UI variants
                return 3;
            case 0x8d84: case 0x8d90: // RGBA16I/UI variants
                return 8;
            case 0x8d86: case 0x8d92: // RGB16I/UI variants
                return 6;
            case 0x8d8a: case 0x8d7e: // RGBA32I/UI variants
                return 16;
            case 0x8d8c: case 0x8d80: // RGB32I/UI variants  
                return 12;
            default:
                // Return a reasonable default for unknown internal formats
                return 4;
        }
    }

    switch(type)
    {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
            return sizeof(uint8_t) * numComponentsForFormat(format);

        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
            return sizeof(uint16_t) * numComponentsForFormat(format);

        case GL_UNSIGNED_INT:
        case GL_INT:
            return sizeof(uint32_t) * numComponentsForFormat(format);

        case GL_FLOAT:
            return sizeof(float) * numComponentsForFormat(format);

        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
            return sizeof(uint8_t);

        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
            return sizeof(uint16_t);

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

        case GL_HALF_FLOAT:
            return sizeof(uint16_t) * numComponentsForFormat(format);

        default:
            fprintf(stderr, "MGL WARNING: sizeForFormatType unknown type 0x%x, format 0x%x\n", type, format);
            return sizeof(uint32_t) * numComponentsForFormat(format);
    }

    return 0;
}

size_t mglPixelTypeDatumBytes(GLenum type)
{
    switch (type) {
        /* Single-byte storage. */
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
            return 1u;

        /* Two-byte storage. */
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_HALF_FLOAT:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
            return 2u;

        /* Four-byte storage. */
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_FLOAT:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_24_8:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
            return 4u;

        /* Eight-byte storage. */
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
            return 8u;

        default:
            return 0u;
    }
}

GLenum verifyInternalFormatType(GLint internalformat, GLenum format, GLenum type)
{
    switch(internalformat)
    {
        // unsized formats
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_STENCIL:
        case GL_RED:
        case GL_RG:
        case GL_RGB:
        case GL_RGBA:
            break;

        // sized formats
        case GL_R8:
        case GL_R8_SNORM:
        case GL_R16:
        case GL_R16_SNORM:
        case GL_RG8:
        case GL_RG8_SNORM:
        case GL_RG16:
        case GL_RG16_SNORM:
        case GL_R3_G3_B2:
        case GL_RGB4:
        case GL_RGB5:
        case GL_RGB8:
        case GL_RGB8_SNORM:
        case GL_RGB10:
        case GL_RGB12:
        case GL_RGB16:
        case GL_RGB16_SNORM:
        case GL_RGBA2:
        case GL_RGBA4:
        case GL_RGB5_A1:
        case GL_RGBA8:
        case GL_RGBA8_SNORM:
        case GL_RGB10_A2:
        case GL_RGB10_A2UI:
        case GL_RGBA12:
        case GL_RGBA16:
        case GL_SRGB:
        case GL_SRGB8:
        case GL_SRGB_ALPHA:
        case GL_SRGB8_ALPHA8:
        case GL_R16F:
        case GL_RG16F:
        case GL_RGB16F:
        case GL_RGBA16F:
        case GL_R32F:
        case GL_RG32F:
        case GL_RGB32F:
        case GL_RGBA32F:
        case GL_R11F_G11F_B10F:
        case GL_RGB9_E5:
        case GL_R8I:
        case GL_R8UI:
        case GL_R16I:
        case GL_R16UI:
        case GL_R32I:
        case GL_R32UI:
        case GL_RG8I:
        case GL_RG8UI:
        case GL_RG16I:
        case GL_RG16UI:
        case GL_RG32I:
        case GL_RG32UI:
        case GL_RGB8I:
        case GL_RGB8UI:
        case GL_RGB16I:
        case GL_RGB16UI:
        case GL_RGB32I:
        case GL_RGB32UI:
        case GL_RGBA8I:
        case GL_RGBA8UI:
        case GL_RGBA16I:
        case GL_RGBA16UI:
        case GL_RGBA32I:
        case GL_RGBA32UI:
            break;

        // compressed types
        case GL_COMPRESSED_RED:
        case GL_COMPRESSED_RG:
        case GL_COMPRESSED_RGB:
        case GL_COMPRESSED_RGBA:
        case GL_COMPRESSED_SRGB:
        case GL_COMPRESSED_SRGB_ALPHA:
        case GL_COMPRESSED_RED_RGTC1:
        case GL_COMPRESSED_SIGNED_RED_RGTC1:
        case GL_COMPRESSED_RG_RGTC2:
        case GL_COMPRESSED_SIGNED_RG_RGTC2:
        case GL_COMPRESSED_RGBA_BPTC_UNORM:
        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
        case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
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
            break;

        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
            if(format != GL_DEPTH_COMPONENT)
                return GL_INVALID_OPERATION;
            break;

        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
            if(format != GL_DEPTH_STENCIL)
                return GL_INVALID_OPERATION;
            break;

        case GL_STENCIL_INDEX1:
        case GL_STENCIL_INDEX4:
        case GL_STENCIL_INDEX8:
        case GL_STENCIL_INDEX16:
            if(format != GL_STENCIL_INDEX)
                return GL_INVALID_OPERATION;
            break;

        default:
            return GL_INVALID_ENUM;
    }

    switch(format)
    {
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
            break;

        default:
            return GL_INVALID_ENUM;
    }

    switch(type)
    {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_FLOAT:
        case GL_HALF_FLOAT:
            break;

        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
            if(format != GL_RGB)
                return GL_INVALID_OPERATION;
            break;

        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
            if (format != GL_RGBA && format != GL_BGRA)
                return GL_INVALID_OPERATION;
            break;
    }

    return true;
}

GLboolean validFormat(GLuint format)
{
    switch(format)
    {
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
            return true;

        default:
            return false;
    }

    return false;
}

GLboolean validFormatType(GLuint format, GLuint type)
{
    RETURN_FALSE_ON_FAILURE(validFormat(format));

    switch(type)
    {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_FLOAT:
        case GL_HALF_FLOAT:
            return true;

        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
            RETURN_FALSE_ON_FAILURE(format == GL_RGB);
            break;

        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
            RETURN_FALSE_ON_FAILURE(format == GL_RGBA || format == GL_BGRA);
            break;

        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
            RETURN_FALSE_ON_FAILURE(format == GL_RGB);
            break;

        default:
            return false;
    }

    return true;
}

GLboolean validInternalFormat(GLint internalformat)
{
    switch(internalformat)
    {
        // unsized formats
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_STENCIL:
        case GL_RED:
        case GL_RG:
        case GL_RGB:
        case GL_RGBA:
            break;

        // sized formats
        case GL_R8:
        case GL_R8_SNORM:
        case GL_R16:
        case GL_R16_SNORM:
        case GL_RG8:
        case GL_RG8_SNORM:
        case GL_RG16:
        case GL_RG16_SNORM:
        case GL_R3_G3_B2:
        case GL_RGB4:
        case GL_RGB5:
        case GL_RGB8:
        case GL_RGB8_SNORM:
        case GL_RGB10:
        case GL_RGB12:
        case GL_RGB16:
        case GL_RGB16_SNORM:
        case GL_RGBA2:
        case GL_RGBA4:
        case GL_RGB5_A1:
        case GL_RGBA8:
        case GL_RGBA8_SNORM:
        case GL_RGB10_A2:
        case GL_RGB10_A2UI:
        case GL_RGBA12:
        case GL_RGBA16:
        case GL_SRGB:
        case GL_SRGB8:
        case GL_SRGB_ALPHA:
        case GL_SRGB8_ALPHA8:
        case GL_R16F:
        case GL_RG16F:
        case GL_RGB16F:
        case GL_RGBA16F:
        case GL_R32F:
        case GL_RG32F:
        case GL_RGB32F:
        case GL_RGBA32F:
        case GL_R11F_G11F_B10F:
        case GL_RGB9_E5:
        case GL_R8I:
        case GL_R8UI:
        case GL_R16I:
        case GL_R16UI:
        case GL_R32I:
        case GL_R32UI:
        case GL_RG8I:
        case GL_RG8UI:
        case GL_RG16I:
        case GL_RG16UI:
        case GL_RG32I:
        case GL_RG32UI:
        case GL_RGB8I:
        case GL_RGB8UI:
        case GL_RGB16I:
        case GL_RGB16UI:
        case GL_RGB32I:
        case GL_RGB32UI:
        case GL_RGBA8I:
        case GL_RGBA8UI:
        case GL_RGBA16I:
        case GL_RGBA16UI:
        case GL_RGBA32I:
        case GL_RGBA32UI:
            break;

        // compressed types
        case GL_COMPRESSED_RED:
        case GL_COMPRESSED_RG:
        case GL_COMPRESSED_RGB:
        case GL_COMPRESSED_RGBA:
        case GL_COMPRESSED_SRGB:
        case GL_COMPRESSED_SRGB_ALPHA:
        case GL_COMPRESSED_RED_RGTC1:
        case GL_COMPRESSED_SIGNED_RED_RGTC1:
        case GL_COMPRESSED_RG_RGTC2:
        case GL_COMPRESSED_SIGNED_RG_RGTC2:
        case GL_COMPRESSED_RGBA_BPTC_UNORM:
        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
        case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
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
            break;

        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
        case GL_STENCIL_INDEX1:
        case GL_STENCIL_INDEX4:
        case GL_STENCIL_INDEX8:
        case GL_STENCIL_INDEX16:
            break;

        default:
            return false;
    }

    return true;
}

#define bitsToBytes(_bits_) ((_bits_ % 8 ? _bits_ / 8 + 1 : _bits_ / 8))
GLuint sizeForInternalFormat(GLenum internalformat, GLenum format, GLenum type)
{
    // return size in bytes
    switch(internalformat)
    {
        case GL_R3_G3_B2:
            return bitsToBytes(8);

        case GL_RGB4:
            return bitsToBytes(12);

        case GL_RGB5:
            return bitsToBytes(15);

        case GL_RGB8:
            return bitsToBytes(24);

        case GL_RGB10:
            return bitsToBytes(30);

        case GL_RGB12:
            /* Metal has no 12-bit format; stored as RGBA16Unorm (16-bit/comp).
             * CTS uses GL_UNSIGNED_SHORT (6 bytes/pixel). */
            return bitsToBytes(48);

        case GL_RGB16:
            return bitsToBytes(48);

        case GL_RGBA2:
            /* GL_RGBA2 nominally uses 2 bits/component, but CTS uploads and
             * reads back with GL_UNSIGNED_SHORT_4_4_4_4 (4 bits/component).
             * Store as 4 bits/component (2 bytes/pixel) to preserve precision. */
            return bitsToBytes(16);

        case GL_RGBA4:
            return bitsToBytes(16);

        case GL_RGB5_A1:
            return bitsToBytes(16);

        case GL_RGBA8:
            return bitsToBytes(32);

        case GL_RGB10_A2:
            return bitsToBytes(32);

        case GL_RGBA12:
            /* Metal has no 12-bit format; stored as RGBA16Unorm (16-bit/comp).
             * CTS uses GL_UNSIGNED_SHORT (8 bytes/pixel). */
            return bitsToBytes(64);

        case GL_RGBA16:
            return bitsToBytes(64);

        case GL_COMPRESSED_RGB:
            return 0;   // return 0 on compressed

        case GL_COMPRESSED_RGBA:
            return 0;   // return 0 on compressed

        case GL_DEPTH_COMPONENT16:
            return bitsToBytes(16);

        case GL_DEPTH_COMPONENT24:
            return bitsToBytes(24);

        case GL_DEPTH_COMPONENT32:
            return bitsToBytes(32);

        case GL_SRGB8:
            return bitsToBytes(24);

        case GL_SRGB_ALPHA:
            return bitsToBytes(32);

        case GL_SRGB8_ALPHA8:
            return bitsToBytes(32);

        case GL_COMPRESSED_SRGB:
            return 0;   // return 0 on compressed

        case GL_COMPRESSED_SRGB_ALPHA:
            return 0;   // return 0 on compressed

        case GL_COMPRESSED_RED:
            return 0;   // return 0 on compressed

        case GL_COMPRESSED_RG:
            return 0;   // return 0 on compressed

        case GL_RGBA32F:
            return bitsToBytes(128);

        case GL_RGB32F:
            return bitsToBytes(96);

        case GL_RGBA16F:
            return bitsToBytes(64);

        case GL_RGB16F:
            return bitsToBytes(48);

        case GL_R11F_G11F_B10F:
            return bitsToBytes(32);

        case GL_RGB9_E5:
            return bitsToBytes(32);

        case GL_RGBA32UI:
            return bitsToBytes(128);

        case GL_RGB32UI:
            return bitsToBytes(96);

        case GL_RGBA16UI:
            return bitsToBytes(64);

        case GL_RGB16UI:
            return bitsToBytes(48);

        case GL_RGBA8UI:
            return bitsToBytes(32);

        case GL_RGB8UI:
            return bitsToBytes(24);

        case GL_RGBA32I:
            return bitsToBytes(128);

        case GL_RGB32I:
            return bitsToBytes(96);

        case GL_RGBA16I:
            return bitsToBytes(64);

        case GL_RGB16I:
            return bitsToBytes(48);

        case GL_RGBA8I:
            return bitsToBytes(32);

        case GL_RGB8I:
            return bitsToBytes(24);

        case GL_DEPTH_COMPONENT32F:
            return bitsToBytes(32);

        case GL_DEPTH32F_STENCIL8:
            return bitsToBytes(40);

        case GL_DEPTH24_STENCIL8:
            return bitsToBytes(32);

        case GL_STENCIL_INDEX1:
            return 1; // bitsToBytes(1);

        case GL_STENCIL_INDEX4:
            return 1; // bitsToBytes(4);

        case GL_STENCIL_INDEX8:
            return bitsToBytes(8);

        case GL_STENCIL_INDEX16:
            return bitsToBytes(16);

        case GL_COMPRESSED_RED_RGTC1:
            return 0;   // return 0 on compressed

        case GL_COMPRESSED_SIGNED_RED_RGTC1:
            return 0;   // return 0 on compressed

        case GL_COMPRESSED_RG_RGTC2:
            return 0;   // return 0 on compressed

        case GL_COMPRESSED_SIGNED_RG_RGTC2:
            return 0;   // return 0 on compressed

        case GL_R8:
            return bitsToBytes(8);

        case GL_R16:
            return bitsToBytes(16);

        case GL_RG8:
            return bitsToBytes(16);

        case GL_RG16:
            return bitsToBytes(32);

        case GL_R16F:
            return bitsToBytes(16);

        case GL_R32F:
            return bitsToBytes(32);

        case GL_RG16F:
            return bitsToBytes(32);

        case GL_RG32F:
            return bitsToBytes(64);

        case GL_R8I:
            return bitsToBytes(8);

        case GL_R8UI:
            return bitsToBytes(8);

        case GL_R16I:
            return bitsToBytes(16);

        case GL_R16UI:
            return bitsToBytes(16);

        case GL_R32I:
            return bitsToBytes(32);

        case GL_R32UI:
            return bitsToBytes(32);

        case GL_RG8I:
            return bitsToBytes(16);

        case GL_RG8UI:
            return bitsToBytes(16);

        case GL_RG16I:
            return bitsToBytes(32);

        case GL_RG16UI:
            return bitsToBytes(32);

        case GL_RG32I:
            return bitsToBytes(64);

        case GL_RG32UI:
            return bitsToBytes(64);

        case GL_R8_SNORM:
            return bitsToBytes(8);

        case GL_RG8_SNORM:
            return bitsToBytes(16);

        case GL_RGB8_SNORM:
            return bitsToBytes(24);

        case GL_RGBA8_SNORM:
            return bitsToBytes(32);

        case GL_R16_SNORM:
            return bitsToBytes(16);

        case GL_RG16_SNORM:
            return bitsToBytes(32);

        case GL_RGB16_SNORM:
            return bitsToBytes(48);

        case GL_RGBA16_SNORM:
            return bitsToBytes(64);

        case GL_RGB10_A2UI:
            return bitsToBytes(32);

        case GL_RGB565:
            return bitsToBytes(16);


        case GL_COMPRESSED_RGBA_BPTC_UNORM:
        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
        case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
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
            return 0;   // return 0 on compressed

        /* Unsized internal formats - must match mtlFormatForGLInternalFormat
         * so the CPU buffer pitch matches the Metal texture pixel size. */
        case GL_RED:
            return bitsToBytes(8);   // R8Unorm
        case GL_RG:
            return bitsToBytes(16);  // RG8Unorm
        case GL_RGB:
        case GL_RGBA:
            return bitsToBytes(32);  // RGBA8Unorm (Metal has no RGB-only)
        case GL_SRGB:
            return bitsToBytes(32);  // RGBA8Unorm_sRGB
        case GL_DEPTH_COMPONENT:
            return bitsToBytes(32);  // Depth32Float
        case GL_DEPTH_STENCIL:
            return bitsToBytes(40);  // Depth32Float_Stencil8
        case GL_STENCIL_INDEX:
            return bitsToBytes(8);   // Stencil8

        /* Legacy luminance/alpha sized formats - must match
         * mtlFormatForGLInternalFormat mappings. */
        case GL_ALPHA8:
        case GL_LUMINANCE8:
            return bitsToBytes(8);   // R8Unorm
        case GL_ALPHA16:
        case GL_LUMINANCE16:
            return bitsToBytes(16);  // R16Unorm
        case GL_ALPHA32F_ARB:
        case GL_LUMINANCE32F_ARB:
            return bitsToBytes(32);  // R32Float
        case GL_ALPHA16F_ARB:
        case GL_LUMINANCE16F_ARB:
            return bitsToBytes(16);  // R16Float
        case GL_LUMINANCE_ALPHA32F_ARB:
            return bitsToBytes(64);  // RG32Float
        case GL_LUMINANCE_ALPHA16F_ARB:
            return bitsToBytes(32);  // RG16Float
        case 0x8045: // GL_LUMINANCE8_ALPHA8
            return bitsToBytes(16);  // RG8Unorm

        default:
            if (internalformat)
            {
                // we didn't get a sized internal format use the internalformat
                // and the src type to figure out a generic size
                return sizeForFormatType(internalformat, type);
            }
            else
            {
                // we didn't get a sized internal format use the src format
                // and the src type to figure out a generic size
                return sizeForFormatType(format, type);
            }
    }

    return 0;
}

GLuint bicountForFormatType(GLenum format, GLenum type, GLenum component)
{
    switch(type)
    {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
            return 8;

        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
            return 16;

        case GL_UNSIGNED_INT:
        case GL_INT:
            return 32;

        case GL_FLOAT:
            return 32;

        case GL_HALF_FLOAT:
            return 16;

        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
            return 8;

        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
            switch(component)
            {
                case GL_RED: return 5;
                case GL_GREEN: return 6;
                case GL_BLUE: return 5;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
            switch(component)
            {
                case GL_RED: return 4;
                case GL_GREEN: return 4;
                case GL_BLUE: return 4;
                case GL_ALPHA: return 4;
            }
            break;


        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
            switch(component)
            {
                case GL_RED: return 5;
                case GL_GREEN: return 5;
                case GL_BLUE: return 5;
                case GL_ALPHA: return 1;
            }
            break;

            return 16;

        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
            return 8;

        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
            switch(component)
            {
                case GL_RED: return 10;
                case GL_GREEN: return 10;
                case GL_BLUE: return 10;
                case GL_ALPHA: return 2;
            }
            break;

        default:
            fprintf(stderr,
                    "MGL WARNING: bicountForFormatType unknown type 0x%x format 0x%x component 0x%x\n",
                    type,
                    format,
                    component);
            return 0;
    }

    return 0;
}

GLuint bitcountForInternalFormat(GLenum internalformat, GLenum component)
{
    // return size in bytes
    switch(internalformat)
    {
        case GL_R3_G3_B2:
            switch(component)
            {
                case GL_RED: return 3;
                case GL_GREEN: return 3;
                case GL_BLUE: return 2;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RGB4:
            switch(component)
            {
                case GL_RED: return 4;
                case GL_GREEN: return 4;
                case GL_BLUE: return 4;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RGB5:
            switch(component)
            {
                case GL_RED: return 5;
                case GL_GREEN: return 5;
                case GL_BLUE: return 5;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RGB8:
            switch(component)
            {
                case GL_RED: return 8;
                case GL_GREEN: return 8;
                case GL_BLUE: return 8;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RGB10:
            switch(component)
            {
                case GL_RED: return 10;
                case GL_GREEN: return 10;
                case GL_BLUE: return 10;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RGB12:
            switch(component)
            {
                case GL_RED: return 16;
                case GL_GREEN: return 16;
                case GL_BLUE: return 16;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RGB16:
            switch(component)
            {
                case GL_RED: return 16;
                case GL_GREEN: return 16;
                case GL_BLUE: return 16;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RGBA2:
            switch(component)
            {
                case GL_RED: return 4;
                case GL_GREEN: return 4;
                case GL_BLUE: return 4;
                case GL_ALPHA: return 4;
            }
            break;

        case GL_RGBA4:
            switch(component)
            {
                case GL_RED: return 4;
                case GL_GREEN: return 4;
                case GL_BLUE: return 4;
                case GL_ALPHA: return 4;
            }
            break;

        case GL_RGB5_A1:
            switch(component)
            {
                case GL_RED: return 5;
                case GL_GREEN: return 5;
                case GL_BLUE: return 5;
                case GL_ALPHA: return 1;
            }
            break;

        case GL_RGBA8:
            switch(component)
            {
                case GL_RED: case GL_GREEN: case GL_BLUE: case GL_ALPHA: return 8;
            }
            break;

        case GL_RGB10_A2:
            switch(component)
            {
                case GL_RED: return 10;
                case GL_GREEN: return 10;
                case GL_BLUE: return 10;
                case GL_ALPHA: return 2;
            }
            break;

        case GL_RGBA12:
            switch(component)
            {
                case GL_RED: case GL_GREEN: case GL_BLUE: case GL_ALPHA: return 16;
            }
            break;

        case GL_RGBA16:
            switch(component)
            {
                case GL_RED: case GL_GREEN: case GL_BLUE: case GL_ALPHA: return 16;
            }
            break;

        case GL_COMPRESSED_RGB:
            return 0;   // return 0 on compressed

        case GL_COMPRESSED_RGBA:
            return 0;   // return 0 on compressed

        case GL_DEPTH_COMPONENT16:
            return component == GL_DEPTH ? 16 : 0;

        case GL_DEPTH_COMPONENT24:
            return component == GL_DEPTH ? 24 : 0;

        case GL_DEPTH_COMPONENT32:
            return component == GL_DEPTH ? 32 : 0;

        case GL_SRGB:
        case GL_SRGB8:
            switch(component)
            {
                case GL_RED: return 8;
                case GL_GREEN: return 8;
                case GL_BLUE: return 8;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_SRGB_ALPHA:
        case GL_SRGB8_ALPHA8:
            switch(component)
            {
                case GL_RED: case GL_GREEN: case GL_BLUE: case GL_ALPHA: return 8;
            }
            break;

        case GL_COMPRESSED_SRGB:
            return 0;   // return 0 on compressed

        case GL_COMPRESSED_SRGB_ALPHA:
            return 0;   // return 0 on compressed

        case GL_COMPRESSED_RED:
            return 0;   // return 0 on compressed

        case GL_COMPRESSED_RG:
            return 0;   // return 0 on compressed

        case GL_RGBA32F:
            switch(component)
            {
                case GL_RED: case GL_GREEN: case GL_BLUE: case GL_ALPHA: return 32;
            }
            break;

        case GL_RGB32F:
            switch(component)
            {
                case GL_RED: return 32;
                case GL_GREEN: return 32;
                case GL_BLUE: return 32;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RGBA16F:
            switch(component)
            {
                case GL_RED: case GL_GREEN: case GL_BLUE: case GL_ALPHA: return 16;
            }
            break;

        case GL_RGB16F:
            switch(component)
            {
                case GL_RED: return 16;
                case GL_GREEN: return 16;
                case GL_BLUE: return 16;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_R11F_G11F_B10F:
            switch(component)
            {
                case GL_RED: return 11;
                case GL_GREEN: return 11;
                case GL_BLUE: return 10;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RGB9_E5:
            switch(component)
            {
                case GL_RED: return 9;
                case GL_GREEN: return 9;
                case GL_BLUE: return 9;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RGBA32UI:
            switch(component)
            {
                case GL_RED: case GL_GREEN: case GL_BLUE: case GL_ALPHA: return 32;
            }
            break;

        case GL_RGB32UI:
            switch(component)
            {
                case GL_RED: return 32;
                case GL_GREEN: return 32;
                case GL_BLUE: return 32;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RGBA16UI:
            switch(component)
            {
                case GL_RED: case GL_GREEN: case GL_BLUE: case GL_ALPHA: return 16;
            }
            break;

        case GL_RGB16UI:
            switch(component)
            {
                case GL_RED: return 16;
                case GL_GREEN: return 16;
                case GL_BLUE: return 16;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RGBA8UI:
            switch(component)
            {
                case GL_RED: case GL_GREEN: case GL_BLUE: case GL_ALPHA: return 8;
            }
            break;

        case GL_RGB8UI:
            switch(component)
            {
                case GL_RED: return 8;
                case GL_GREEN: return 8;
                case GL_BLUE: return 8;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RGBA32I:
            switch(component)
            {
                case GL_RED: case GL_GREEN: case GL_BLUE: case GL_ALPHA: return 32;
            }
            break;

        case GL_RGB32I:
            switch(component)
            {
                case GL_RED: return 32;
                case GL_GREEN: return 32;
                case GL_BLUE: return 32;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RGBA16I:
            switch(component)
            {
                case GL_RED: case GL_GREEN: case GL_BLUE: case GL_ALPHA: return 16;
            }
            break;

        case GL_RGB16I:
            switch(component)
            {
                case GL_RED: return 16;
                case GL_GREEN: return 16;
                case GL_BLUE: return 16;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RGBA8I:
            switch(component)
            {
                case GL_RED: case GL_GREEN: case GL_BLUE: case GL_ALPHA: return 8;
            }
            break;

        case GL_RGB8I:
            switch(component)
            {
                case GL_RED: return 8;
                case GL_GREEN: return 8;
                case GL_BLUE: return 8;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_DEPTH_COMPONENT32F:
            return component == GL_DEPTH ? 32 : 0;

        case GL_DEPTH32F_STENCIL8:
            switch(component)
            {
                case GL_DEPTH: return 32;
                case GL_STENCIL: return 8;
            }
            break;

        case GL_DEPTH24_STENCIL8:
            switch(component)
            {
                case GL_DEPTH: return 24;
                case GL_STENCIL: return 8;
            }
            break;

        case GL_STENCIL_INDEX1:
            return component == GL_STENCIL ? 1 : 0;

        case GL_STENCIL_INDEX4:
            return component == GL_STENCIL ? 4 : 0;

        case GL_STENCIL_INDEX8:
            return component == GL_STENCIL ? 8 : 0;

        case GL_STENCIL_INDEX16:
            return component == GL_STENCIL ? 16 : 0;

        case GL_COMPRESSED_RED_RGTC1:
            return 0;   // return 0 on compressed

        case GL_COMPRESSED_SIGNED_RED_RGTC1:
            return 0;   // return 0 on compressed

        case GL_COMPRESSED_RG_RGTC2:
            return 0;   // return 0 on compressed

        case GL_COMPRESSED_SIGNED_RG_RGTC2:
            return 0;   // return 0 on compressed

        case GL_R8:
            switch(component)
            {
                case GL_RED: return 8;
                case GL_GREEN: return 0;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_R16:
            switch(component)
            {
                case GL_RED: return 16;
                case GL_GREEN: return 0;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RG8:
            switch(component)
            {
                case GL_RED: return 8;
                case GL_GREEN: return 8;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RG16:
            switch(component)
            {
                case GL_RED: return 16;
                case GL_GREEN: return 16;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_R16F:
            switch(component)
            {
                case GL_RED: return 16;
                case GL_GREEN: return 0;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_R32F:
            switch(component)
            {
                case GL_RED: return 32;
                case GL_GREEN: return 0;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RG16F:
            switch(component)
            {
                case GL_RED: return 16;
                case GL_GREEN: return 16;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RG32F:
            switch(component)
            {
                case GL_RED: return 32;
                case GL_GREEN: return 32;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_R8I:
            switch(component)
            {
                case GL_RED: return 8;
                case GL_GREEN: return 0;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_R8UI:
            switch(component)
            {
                case GL_RED: return 8;
                case GL_GREEN: return 0;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_R16I:
            switch(component)
            {
                case GL_RED: return 16;
                case GL_GREEN: return 0;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_R16UI:
            switch(component)
            {
                case GL_RED: return 16;
                case GL_GREEN: return 0;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_R32I:
            switch(component)
            {
                case GL_RED: return 32;
                case GL_GREEN: return 0;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_R32UI:
            switch(component)
            {
                case GL_RED: return 32;
                case GL_GREEN: return 0;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RG8I:
            switch(component)
            {
                case GL_RED: return 8;
                case GL_GREEN: return 8;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RG8UI:
            switch(component)
            {
                case GL_RED: return 8;
                case GL_GREEN: return 8;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RG16I:
            switch(component)
            {
                case GL_RED: return 16;
                case GL_GREEN: return 16;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RG16UI:
            switch(component)
            {
                case GL_RED: return 16;
                case GL_GREEN: return 16;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RG32I:
            switch(component)
            {
                case GL_RED: return 32;
                case GL_GREEN: return 32;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RG32UI:
            switch(component)
            {
                case GL_RED: return 32;
                case GL_GREEN: return 32;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_R8_SNORM:
            switch(component)
            {
                case GL_RED: return 8;
                case GL_GREEN: return 0;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RG8_SNORM:
            switch(component)
            {
                case GL_RED: return 8;
                case GL_GREEN: return 8;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RGB8_SNORM:
            switch(component)
            {
                case GL_RED: return 8;
                case GL_GREEN: return 8;
                case GL_BLUE: return 8;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RGBA8_SNORM:
            return 8;

        case GL_R16_SNORM:
            switch(component)
            {
                case GL_RED: return 16;
                case GL_GREEN: return 0;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RG16_SNORM:
            switch(component)
            {
                case GL_RED: return 16;
                case GL_GREEN: return 16;
                case GL_BLUE: return 0;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RGB16_SNORM:
            switch(component)
            {
                case GL_RED: return 16;
                case GL_GREEN: return 16;
                case GL_BLUE: return 16;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_RGBA16_SNORM:
            return 16;

        case GL_RGB10_A2UI:
            switch(component)
            {
                case GL_RED: return 10;
                case GL_GREEN: return 10;
                case GL_BLUE: return 10;
                case GL_ALPHA: return 2;
            }
            break;

        case GL_RGB565:
            switch(component)
            {
                case GL_RED: return 5;
                case GL_GREEN: return 6;
                case GL_BLUE: return 5;
                case GL_ALPHA: return 0;
            }
            break;

        case GL_COMPRESSED_RGBA_BPTC_UNORM:
        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
        case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
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
            return 0;   // return 0 on compressed

        default:
            return 0;
    }

    return 0;
}

GLenum internalFormatForGLFormatType(GLenum format, GLenum type)
{
    switch(type)
    {
        case GL_UNSIGNED_BYTE:
            switch(format)
            {
                case GL_RED: return GL_R8;
                case GL_RG: return GL_RG8;
                case GL_RGB: return GL_RGB8;
                case GL_BGR: return GL_RGB8;  /* BGR treated as RGB */
                case GL_RGBA: return GL_RGBA8;
                case GL_BGRA: return GL_RGBA8;  /* BGRA treated as RGBA */
                case GL_DEPTH_COMPONENT: return GL_DEPTH_COMPONENT16;
                case GL_STENCIL_INDEX: return GL_STENCIL_INDEX8;
                default:
                    return 0;
            }
            break;

        case GL_BYTE:
            switch(format)
            {
                case GL_RED: return GL_R8_SNORM;
                case GL_RG: return GL_RG8_SNORM;
                case GL_RGB: return GL_RGB8_SNORM;
                case GL_RGBA: return GL_RGBA8_SNORM;
                case GL_DEPTH_COMPONENT: return GL_DEPTH_COMPONENT16;
                case GL_STENCIL_INDEX: return GL_STENCIL_INDEX8;
                default:
                    return 0;
            }
            break;

        case GL_UNSIGNED_SHORT:
            switch(format)
            {
                case GL_RED: return GL_R16;
                case GL_RG: return GL_RG16;
                case GL_RGB: return GL_RGB16;
                case GL_RGBA: return GL_RGBA16;
                case GL_DEPTH_COMPONENT: return GL_DEPTH_COMPONENT16;
                case GL_STENCIL_INDEX: return GL_STENCIL_INDEX8;
                default:
                    return 0;
            }
            break;

        case GL_SHORT:
            switch(format)
            {
                case GL_RED: return GL_R16_SNORM;
                case GL_RG: return GL_RG16_SNORM;
                case GL_RGB: return GL_RGB16_SNORM;
                case GL_RGBA: return GL_RGBA16_SNORM;
                case GL_DEPTH_COMPONENT: return GL_DEPTH_COMPONENT16;
                case GL_STENCIL_INDEX: return GL_STENCIL_INDEX8;
                default:
                    return 0;
            }
            break;

        case GL_UNSIGNED_INT:
            switch(format)
            {
                /* Non-integer formats with integer types: resolve to float
                 * sized formats, NOT integer formats. GL_RED/GL_RG/GL_RGB/
                 * GL_RGBA are non-integer (UNORM) formats; using GL_UNSIGNED_INT
                 * as the type does NOT make the texture integer. Only
                 * GL_*_INTEGER formats create integer textures. */
                case GL_RED: return GL_R32F;
                case GL_RG: return GL_RG32F;
                case GL_RGB: return GL_RGB32F;
                case GL_RGBA: return GL_RGBA32F;
                case GL_DEPTH_COMPONENT: return GL_DEPTH_COMPONENT24;
                case GL_STENCIL_INDEX: return GL_STENCIL_INDEX8;
                default:
                    return 0;
            }
            break;

        case GL_INT:
            switch(format)
            {
                /* See comment above: non-integer formats stay non-integer. */
                case GL_RED: return GL_R32F;
                case GL_RG: return GL_RG32F;
                case GL_RGB: return GL_RGB32F;
                case GL_RGBA: return GL_RGBA32F;
                case GL_DEPTH_COMPONENT: return GL_DEPTH_COMPONENT32F;
                case GL_STENCIL_INDEX: return GL_STENCIL_INDEX8;
                default:
                    return 0;
            }
            break;

        case GL_FLOAT:
            switch(format)
            {
                case GL_RED: return GL_R32F;
                case GL_RG: return GL_RG32F;
                case GL_RGB: return GL_RGB32F;
                case GL_RGBA: return GL_RGBA32F;
                case GL_DEPTH_COMPONENT: return GL_DEPTH_COMPONENT32F;
                case GL_DEPTH_STENCIL: return GL_DEPTH32F_STENCIL8;
                case GL_STENCIL_INDEX: return GL_STENCIL_INDEX8;
                default:
                    return 0;
            }
            break;

        case GL_HALF_FLOAT:
            switch(format)
            {
                case GL_RED: return GL_R16F;
                case GL_RG: return GL_RG16F;
                case GL_RGB: return GL_RGB16F;
                case GL_RGBA: return GL_RGBA16F;
                case GL_DEPTH_COMPONENT: return GL_DEPTH_COMPONENT32F;
                case GL_STENCIL_INDEX: return GL_STENCIL_INDEX8;
                default:
                    return 0;
            }
            break;

        case GL_UNSIGNED_BYTE_3_3_2:
            return 0;

        case GL_UNSIGNED_BYTE_2_3_3_REV:
            return 0;

        case GL_UNSIGNED_SHORT_5_6_5:
            return GL_RGB565;

        case GL_UNSIGNED_SHORT_5_6_5_REV:
            return 0;

        case GL_UNSIGNED_SHORT_4_4_4_4:
            return GL_RGBA4;

        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
            return 0;

        case GL_UNSIGNED_INT_8_8_8_8:
            return GL_RGBA8;

        case GL_UNSIGNED_INT_8_8_8_8_REV:
            return GL_RGBA8;

        case GL_UNSIGNED_INT_24_8:
            if (format == GL_DEPTH_STENCIL)
                return GL_DEPTH24_STENCIL8;
            return 0;

        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
            if (format == GL_DEPTH_STENCIL)
                return GL_DEPTH32F_STENCIL8;
            return 0;

        default:
            fprintf(stderr,
                    "MGL WARNING: internalFormatForGLFormatType unknown type 0x%x format 0x%x\n",
                    type,
                    format);
            return 0;
    }
}

uint32_t mtlFormatForGLInternalFormat(GLenum internal_format)
{
    switch(internal_format)
    {
        case GL_RGB4:
        case GL_RGB5:
            return MGLPixelFormatRGBA8Unorm;  // Upconvert to RGBA8
            
        case GL_RGB8:
            return MGLPixelFormatRGBA8Unorm;  // Metal doesn't have RGB-only formats

        case GL_RGB10:
            return MGLPixelFormatRGB10A2Unorm;  // 10-bit per channel

        case GL_RGB12:
        case GL_RGB16:
            return MGLPixelFormatRGBA16Unorm;  // 16-bit per channel
            
        case GL_RGBA2:
        case GL_RGBA4:
        case GL_RGB5_A1:
            return MGLPixelFormatRGBA8Unorm;  // Upconvert to avoid ABGR4/BGR5A1 bit order mismatch

        case GL_RGBA8:
            return MGLPixelFormatRGBA8Unorm;    // working format
            //return MGLPixelFormatBGRA8Unorm;    // working format

        case GL_R3_G3_B2:
            return MGLPixelFormatRGBA8Unorm;    // Upconvert to RGBA8

        case GL_ALPHA8UI_EXT:
            return MGLPixelFormatR8Uint;        // Map Alpha Integer to Red Integer (best effort)

        case GL_RGB10_A2:
            return MGLPixelFormatRGB10A2Unorm;    // working format

        case GL_RGBA12:
            return MGLPixelFormatRGBA16Unorm;  // Upconvert 12-bit to 16-bit

        case GL_RGBA16:
            return MGLPixelFormatRGBA16Unorm;    // working format

        case GL_COMPRESSED_RGB:
            if (__builtin_available(macOS 11.0, *)) {
                return MGLPixelFormatETC2_RGB8;
            } else {
                // Fallback on earlier versions
                return MGLPixelFormatInvalid;
            }

        case GL_COMPRESSED_RGBA:
            if (__builtin_available(macOS 11.0, *)) {
                return MGLPixelFormatEAC_RGBA8;
            } else {
                // Fallback on earlier versions
                return MGLPixelFormatInvalid;
            }

        case GL_DEPTH_COMPONENT16:
            return MGLPixelFormatDepth16Unorm;

        case GL_DEPTH_COMPONENT24:
            // Apple Silicon doesn't support 24-bit depth, use 32-bit float instead.
            // (Intel Mac MGLPixelFormatDepth24Unorm_Stencil8 exists, but a
            // depth-only 24-bit format does not; Depth32Float is the portable
            // choice.  CPU shadow buffers still store 24-bit unorm and the
            // upload path normalizes uint→float when filling the Metal texture.)
            return MGLPixelFormatDepth32Float;

        case GL_DEPTH_COMPONENT32:
            // GL spec defines DEPTH_COMPONENT32 as a 32-bit unorm format
            // (values normalized to [0,1]); mapping it to Depth32Float is
            // semantically correct — both represent normalized depth, and
            // Metal has no 32-bit unorm depth format.  As with the 24-bit
            // case, CPU shadow buffers store uint32 and the upload path
            // (rendering.c) performs the uint→float normalization.
            return MGLPixelFormatDepth32Float;

        case GL_SRGB:
            return MGLPixelFormatRGBA8Unorm_sRGB;  // Upconvert to RGBA8 sRGB

        case GL_SRGB8:
            return MGLPixelFormatRGBA8Unorm_sRGB;

        case GL_SRGB_ALPHA:
            return MGLPixelFormatRGBA8Unorm_sRGB;  // Upconvert to RGBA8 sRGB

        case GL_SRGB8_ALPHA8:
            return MGLPixelFormatRGBA8Unorm_sRGB;

        case GL_COMPRESSED_SRGB:
            if (__builtin_available(macOS 11.0, *)) {
                return MGLPixelFormatETC2_RGB8_sRGB;
            } else {
                return MGLPixelFormatInvalid;
            }

        case GL_COMPRESSED_SRGB_ALPHA:
            return MGLPixelFormatRGBA8Unorm_sRGB;  // Decompress to RGBA8 sRGB

        case GL_COMPRESSED_RED:
            if (__builtin_available(macOS 11.0, *)) {
                return MGLPixelFormatEAC_R11Unorm;
            } else {
                return MGLPixelFormatInvalid;
            }

        case GL_COMPRESSED_RG:
            if (__builtin_available(macOS 11.0, *)) {
                return MGLPixelFormatEAC_RG11Unorm;
            } else {
                return MGLPixelFormatInvalid;
            }

        case GL_RGBA32F:
            return MGLPixelFormatRGBA32Float;

        case GL_RGB32F:
            return MGLPixelFormatRGBA32Float;

        case GL_RGBA16F:
            return MGLPixelFormatRGBA16Float;

        case GL_RGB16F:
            return MGLPixelFormatRGBA16Float;

        case GL_R11F_G11F_B10F:
            return MGLPixelFormatRG11B10Float;

        case GL_RGB9_E5:
            return MGLPixelFormatRGB9E5Float;

        case GL_RGBA32UI:
            return MGLPixelFormatRGBA32Uint;

        case GL_RGB32UI:
            return MGLPixelFormatRGBA32Uint;

        case GL_RGBA16UI:
            return MGLPixelFormatRGBA16Uint;

        case GL_RGB16UI:
            return MGLPixelFormatRGBA16Uint;

        case GL_RGBA8UI:
            return MGLPixelFormatRGBA8Uint;

        case GL_RGB8UI:
            return MGLPixelFormatRGBA8Uint;

        case GL_RGBA32I:
            return MGLPixelFormatRGBA32Sint;

        case GL_RGB32I:
            return MGLPixelFormatRGBA32Sint;

        case GL_RGBA16I:
            return MGLPixelFormatRGBA16Sint;

        case GL_RGB16I:
            return MGLPixelFormatRGBA16Sint;

        case GL_RGBA8I:
            return MGLPixelFormatRGBA8Sint;

        case GL_RGB8I:
            return MGLPixelFormatRGBA8Sint;

        case GL_DEPTH_COMPONENT32F:
            return MGLPixelFormatDepth32Float;

        case GL_DEPTH32F_STENCIL8:
            return MGLPixelFormatDepth32Float_Stencil8;

        case GL_DEPTH24_STENCIL8:
            // MGLPixelFormatX24_Stencil8 (262) is NOT supported on Apple Silicon
            // Use Depth32Float_Stencil8 instead
            return MGLPixelFormatDepth32Float_Stencil8;

        case GL_STENCIL_INDEX1:
            // Metal only supports an 8-bit stencil format
            // (MGLPixelFormatStencil8); 1-bit stencil has no Metal equivalent.
            // Returning Invalid makes glTexStorage2D reject this format.
            // (glTexImage2D callers could be relaxed to fall back to
            // STENCIL_INDEX8, but that is a behavior change left for later.)
            return MGLPixelFormatInvalid;

        case GL_STENCIL_INDEX4:
            // Metal only supports 8-bit stencil; 4-bit has no equivalent.
            // Returning Invalid makes glTexStorage2D reject this format.
            return MGLPixelFormatInvalid;

        case GL_STENCIL_INDEX8:
            return MGLPixelFormatStencil8;

        case GL_STENCIL_INDEX16:
            // Metal only supports 8-bit stencil; 16-bit has no equivalent.
            // Returning Invalid makes glTexStorage2D reject this format.
            return MGLPixelFormatInvalid;

        case GL_COMPRESSED_RED_RGTC1:
            return MGLPixelFormatBC4_RUnorm;

        case GL_COMPRESSED_SIGNED_RED_RGTC1:
            return MGLPixelFormatBC4_RSnorm;

        case GL_COMPRESSED_RG_RGTC2:
            return MGLPixelFormatBC5_RGUnorm;

        case GL_COMPRESSED_SIGNED_RG_RGTC2:
            return MGLPixelFormatBC5_RGSnorm;
        /* S3TC/DXT (GL_EXT_texture_compression_s3tc) — BC1/BC2/BC3.
         * Available on macOS 10.11+; native on Apple Silicon. */
        case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
            return MGLPixelFormatBC1_RGBA;
        case 0x8c4c: /* GL_COMPRESSED_SRGB_S3TC_DXT1_EXT */
        case 0x8c4d: /* GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT */
            return MGLPixelFormatBC1_RGBA_sRGB;
        case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
            return MGLPixelFormatBC2_RGBA;
        case 0x8c4e: /* GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT */
            return MGLPixelFormatBC2_RGBA_sRGB;
        case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:
            return MGLPixelFormatBC3_RGBA;
        case 0x8c4f: /* GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT */
            return MGLPixelFormatBC3_RGBA_sRGB;

        /* ASTC LDR (GL_KHR_texture_compression_astc_ldr) — macOS 11+. */
        case GL_COMPRESSED_RGBA_ASTC_4x4_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_4x4_LDR : MGLPixelFormatInvalid;
        case GL_COMPRESSED_RGBA_ASTC_5x4_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_5x4_LDR : MGLPixelFormatInvalid;
        case GL_COMPRESSED_RGBA_ASTC_5x5_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_5x5_LDR : MGLPixelFormatInvalid;
        case GL_COMPRESSED_RGBA_ASTC_6x5_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_6x5_LDR : MGLPixelFormatInvalid;
        case GL_COMPRESSED_RGBA_ASTC_6x6_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_6x6_LDR : MGLPixelFormatInvalid;
        case GL_COMPRESSED_RGBA_ASTC_8x5_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_8x5_LDR : MGLPixelFormatInvalid;
        case GL_COMPRESSED_RGBA_ASTC_8x6_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_8x6_LDR : MGLPixelFormatInvalid;
        case GL_COMPRESSED_RGBA_ASTC_8x8_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_8x8_LDR : MGLPixelFormatInvalid;
        case GL_COMPRESSED_RGBA_ASTC_10x5_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_10x5_LDR : MGLPixelFormatInvalid;
        case GL_COMPRESSED_RGBA_ASTC_10x6_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_10x6_LDR : MGLPixelFormatInvalid;
        case GL_COMPRESSED_RGBA_ASTC_10x8_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_10x8_LDR : MGLPixelFormatInvalid;
        case GL_COMPRESSED_RGBA_ASTC_10x10_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_10x10_LDR : MGLPixelFormatInvalid;
        case GL_COMPRESSED_RGBA_ASTC_12x10_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_12x10_LDR : MGLPixelFormatInvalid;
        case GL_COMPRESSED_RGBA_ASTC_12x12_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_12x12_LDR : MGLPixelFormatInvalid;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_4x4_sRGB : MGLPixelFormatInvalid;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_5x4_sRGB : MGLPixelFormatInvalid;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_5x5_sRGB : MGLPixelFormatInvalid;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_6x5_sRGB : MGLPixelFormatInvalid;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_6x6_sRGB : MGLPixelFormatInvalid;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_8x5_sRGB : MGLPixelFormatInvalid;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_8x6_sRGB : MGLPixelFormatInvalid;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_8x8_sRGB : MGLPixelFormatInvalid;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_10x5_sRGB : MGLPixelFormatInvalid;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_10x6_sRGB : MGLPixelFormatInvalid;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_10x8_sRGB : MGLPixelFormatInvalid;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_10x10_sRGB : MGLPixelFormatInvalid;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_12x10_sRGB : MGLPixelFormatInvalid;
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:
            return __builtin_available(macOS 11.0, *) ? MGLPixelFormatASTC_12x12_sRGB : MGLPixelFormatInvalid;

        case GL_R8:
            return MGLPixelFormatR8Unorm;

        case GL_R16:
            return MGLPixelFormatR16Unorm;

        case GL_RG8:
            return MGLPixelFormatRG8Unorm;

        case GL_RG16:
            return MGLPixelFormatRG16Unorm;

        case GL_R16F:
            return MGLPixelFormatR16Float;

        case GL_R32F:
            return MGLPixelFormatR32Float;

        case GL_RG16F:
            return MGLPixelFormatRG16Float;

        case GL_RG32F:
            return MGLPixelFormatRG32Float;

        case GL_R8I:
            return MGLPixelFormatR8Sint;

        case GL_R8UI:
            return MGLPixelFormatR8Uint;

        case GL_R16I:
            return MGLPixelFormatR16Sint;

        case GL_R16UI:
            return MGLPixelFormatR16Uint;

        case GL_R32I:
            return MGLPixelFormatR32Sint;

        case GL_R32UI:
            return MGLPixelFormatR32Uint;

        case GL_RG8I:
            return MGLPixelFormatRG8Sint;

        case GL_RG8UI:
            return MGLPixelFormatRG8Uint;

        case GL_RG16I:
            return MGLPixelFormatRG16Sint;

        case GL_RG16UI:
            return MGLPixelFormatRG16Uint;

        case GL_RG32I:
            return MGLPixelFormatRG32Sint;

        case GL_RG32UI:
            return MGLPixelFormatRG32Uint;

        case GL_R8_SNORM:
            return MGLPixelFormatR8Snorm;

        case GL_RG8_SNORM:
            return MGLPixelFormatRG8Snorm;

        case GL_RGB8_SNORM:
            return MGLPixelFormatRGBA8Snorm;

        case GL_RGBA8_SNORM:
            return MGLPixelFormatRGBA8Snorm;

        case GL_R16_SNORM:
            return MGLPixelFormatR16Snorm;

        case GL_RG16_SNORM:
            return MGLPixelFormatRG16Snorm;

        case GL_RGB16_SNORM:
            return MGLPixelFormatRGBA16Snorm;

        case GL_RGBA16_SNORM:
            return MGLPixelFormatRGBA16Snorm;

        case GL_RGB10_A2UI:
            return MGLPixelFormatRGB10A2Uint;

        case GL_RGB565:
            /* MGLPixelFormatB5G6R5Unorm places B in the high bits, but GL
             * UNSIGNED_SHORT_5_6_5 places R in the high bits — sampling would
             * swap R and B.  Back GL_RGB565 with RGBA8Unorm instead and let
             * mglCreateRGBA8ExpandedUpload rearrange channels on the CPU.
             * (mglTextureInternalFormatNeedsRGBA8Expansion returns true for
             * GL_RGB565 to drive that expansion path.) */
            return MGLPixelFormatRGBA8Unorm;

        // Legacy unsized formats - map to sized equivalents
        case GL_RED:
            return MGLPixelFormatR8Unorm;
            
        case GL_RGBA:
            return MGLPixelFormatRGBA8Unorm;
            
        case GL_RGB:
            return MGLPixelFormatRGBA8Unorm;  // No RGB-only format in Metal
            
        // Legacy luminance/alpha formats - map to R/RG
        case GL_ALPHA8:
        case GL_LUMINANCE8:
            return MGLPixelFormatR8Unorm;
            
        case GL_ALPHA16:
        case GL_LUMINANCE16:
            return MGLPixelFormatR16Unorm;
            
        case GL_ALPHA32F_ARB:
        case GL_LUMINANCE32F_ARB:
            return MGLPixelFormatR32Float;
            
        case GL_ALPHA16F_ARB:
        case GL_LUMINANCE16F_ARB:
            return MGLPixelFormatR16Float;
            
        case GL_LUMINANCE_ALPHA32F_ARB:
            return MGLPixelFormatRG32Float;
            
        case GL_LUMINANCE_ALPHA16F_ARB:
            return MGLPixelFormatRG16Float;
            
        case 0x8045: // GL_LUMINANCE8_ALPHA8
            return MGLPixelFormatRG8Unorm;
            
        // Note: 0x8048 (GL_LUMINANCE16_ALPHA16) already handled by GL_LUMINANCE16 case above
        // due to incorrect macro definition
            
        // sRGB R/RG formats
        case GL_SR8_EXT:
            return MGLPixelFormatR8Unorm_sRGB;
            
        case GL_SRG8_EXT:
            return MGLPixelFormatRG8Unorm_sRGB;

        case GL_COMPRESSED_RGBA_BPTC_UNORM:
            return MGLPixelFormatBC7_RGBAUnorm;

        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
            return MGLPixelFormatBC7_RGBAUnorm_sRGB;

        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
            return MGLPixelFormatBC6H_RGBFloat;

        case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
            return MGLPixelFormatBC6H_RGBUfloat;

        case GL_COMPRESSED_RGB8_ETC2:
            if (__builtin_available(macOS 11.0, *)) {
                return MGLPixelFormatETC2_RGB8;
            } else {
                // Fallback on earlier versions
                return MGLPixelFormatInvalid;
            }

        case GL_COMPRESSED_SRGB8_ETC2:
            if (__builtin_available(macOS 11.0, *)) {
                return MGLPixelFormatETC2_RGB8_sRGB;
            } else {
                // Fallback on earlier versions
                return MGLPixelFormatInvalid;
            }

        case GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2:
            if (__builtin_available(macOS 11.0, *)) {
                return MGLPixelFormatETC2_RGB8A1;
            } else {
                // Fallback on earlier versions
                return MGLPixelFormatInvalid;
            }

        case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2:
            if (__builtin_available(macOS 11.0, *)) {
                return MGLPixelFormatETC2_RGB8A1_sRGB;
            } else {
                // Fallback on earlier versions
                return MGLPixelFormatInvalid;
            }

        case GL_COMPRESSED_RGBA8_ETC2_EAC:
            if (__builtin_available(macOS 11.0, *)) {
                return MGLPixelFormatEAC_RGBA8;
            } else {
                // Fallback on earlier versions
                return MGLPixelFormatInvalid;
            }

        case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:
            if (__builtin_available(macOS 11.0, *)) {
                return MGLPixelFormatEAC_RGBA8_sRGB;
            } else {
                // Fallback on earlier versions
                return MGLPixelFormatInvalid;
            }

        case GL_COMPRESSED_R11_EAC:
            if (__builtin_available(macOS 11.0, *)) {
                return MGLPixelFormatEAC_R11Unorm;
            } else {
                // Fallback on earlier versions
                return MGLPixelFormatInvalid;
            }

        case GL_COMPRESSED_SIGNED_R11_EAC:
            if (__builtin_available(macOS 11.0, *)) {
                return MGLPixelFormatEAC_R11Snorm;
            } else {
                // Fallback on earlier versions
                return MGLPixelFormatInvalid;
            }

        case GL_COMPRESSED_RG11_EAC:
            if (__builtin_available(macOS 11.0, *)) {
                return MGLPixelFormatEAC_RG11Unorm;
            } else {
                // Fallback on earlier versions
                return MGLPixelFormatInvalid;
            }

        case GL_COMPRESSED_SIGNED_RG11_EAC:
            if (__builtin_available(macOS 11.0, *)) {
                return MGLPixelFormatEAC_RG11Snorm;
            } else {
                // Fallback on earlier versions
                return MGLPixelFormatInvalid;
            }

        // Unsized base formats - virglrenderer may pass these
        case GL_RG:
            return MGLPixelFormatRG8Unorm;
            
        case GL_DEPTH_COMPONENT:
            return MGLPixelFormatDepth32Float;
            
        case GL_DEPTH_STENCIL:
            return MGLPixelFormatDepth32Float_Stencil8;
            
        case GL_STENCIL_INDEX:
            return MGLPixelFormatStencil8;

        // Additional integer formats (alternate enum values used by some implementations)
        case 0x8d72: // GL_ALPHA32UI_EXT
            return MGLPixelFormatR32Uint;
        case 0x8d75: // alternate GL_RGB8I
            return MGLPixelFormatRGBA8Sint;
        case 0x8d78: // alternate GL_RGBA8UI
            return MGLPixelFormatRGBA8Uint;
        case 0x8d7a: // alternate GL_RGB8UI
            return MGLPixelFormatRGBA8Uint;
        case 0x8d7b: // GL_ALPHA8I_EXT
            return MGLPixelFormatR8Sint;
        // case 0x8d7e: // GL_ALPHA8UI_EXT - Duplicate
        //    return MGLPixelFormatR8Uint;
        case 0x8d80: // GL_LUMINANCE8UI_EXT
            return MGLPixelFormatR8Uint;
        case 0x8d81: // GL_ALPHA32I_EXT
            return MGLPixelFormatR32Sint;
        case 0x8d84: // alternate GL_RGBA16I
            return MGLPixelFormatRGBA16Sint;
        case 0x8d86: // alternate GL_RGB16I
            return MGLPixelFormatRGBA16Sint;
        case 0x8d87: // GL_ALPHA16I_EXT
            return MGLPixelFormatR16Sint;
        case 0x8d8a: // alternate GL_RGBA32I
            return MGLPixelFormatRGBA32Sint;
        case 0x8d8c: // alternate GL_RGB32I
            return MGLPixelFormatRGBA32Sint;
        
        // SNORM formats
        case 0x9014: // GL_ALPHA8_SNORM
            return MGLPixelFormatR8Snorm;
        case 0x9016: // GL_LUMINANCE8_ALPHA8_SNORM
            return MGLPixelFormatRG8Snorm;
        case 0x9018: // GL_ALPHA16_SNORM
            return MGLPixelFormatR16Snorm;
        case 0x901a: // GL_LUMINANCE16_ALPHA16_SNORM
            return MGLPixelFormatRG16Snorm;
        case 0x8d8d: // GL_ALPHA32I_EXT
            return MGLPixelFormatR32Sint;
        case 0x8d90: // alternate GL_RGBA16UI
            return MGLPixelFormatRGBA16Uint;
        case 0x8d92: // alternate GL_RGB16UI
            return MGLPixelFormatRGBA16Uint;
        case 0x8d93: // GL_ALPHA16UI_EXT
            return MGLPixelFormatR16Uint;

        default:
            // Unknown formats - likely Mesa/Gallium internal format enums or capability probes
            // Return Invalid to indicate format not supported (don't use fallback for probes)
            // Only warn for formats that look like real GL formats (not obvious enum values)
            if (internal_format >= 0x1 && internal_format <= 0x2000) {
                // Low values might be legacy GL formats - warn about these
                static unsigned warned_formats[64] = {0};
                static int warned_count = 0;
                int already_warned = 0;
                for (int i = 0; i < warned_count && i < 64; i++) {
                    if (warned_formats[i] == internal_format) { already_warned = 1; break; }
                }
                if (!already_warned && warned_count < 64) {
                    warned_formats[warned_count++] = internal_format;
                    fprintf(stderr, "MGL WARNING: mtlFormatForGLInternalFormat unknown format 0x%x\n", internal_format);
                }
            }
            // For 0x8Dxx and 0x90xx ranges - these are often internal/capability probes
            // Silently return Invalid to indicate "not supported"
            return MGLPixelFormatInvalid;
    }

    return MGLPixelFormatInvalid;
}

GLboolean mglIsColorRenderableInternalFormat(GLint internalformat)
{
    /* GL 4.6 required color-renderable formats (Table 8.11).
     * Matches the list used by the CTS packed_pixels isFBOImageAttachValid
     * for non-ES contexts.  Unsized base formats are canonicalised first. */
    switch (internalformat)
    {
        /* Unsized base formats - canonicalise to sized equivalents.
         * GL_RED->R8, GL_RG->RG8, GL_RGB->RGB8, GL_RGBA->RGBA8 are all
         * color-renderable.  GL_SRGB is not (SRGB8 is not in the required
         * list), but GL_SRGB_ALPHA->SRGB8_ALPHA8 is. */
        case GL_RED: return GL_TRUE;
        case GL_RG: return GL_TRUE;
        case GL_RGB: return GL_TRUE;
        case GL_RGBA: return GL_TRUE;
        case GL_SRGB: return GL_FALSE;
        case GL_SRGB_ALPHA: return GL_TRUE;

        /* Required color-renderable sized formats (CTS colorRenderableFrmats). */
        case GL_RGBA32F:
        case GL_RGBA32I:
        case GL_RGBA32UI:
        case GL_RGBA16:
        case GL_RGBA16F:
        case GL_RGBA16I:
        case GL_RGBA16UI:
        case GL_RGBA8:
        case GL_RGBA8I:
        case GL_RGBA8UI:
        case GL_SRGB8_ALPHA8:
        case GL_RGB10_A2:
        case GL_RGB10_A2UI:
        case GL_RGB5_A1:
        case GL_RGBA4:
        case GL_R11F_G11F_B10F:
        case GL_RGB565:
        case GL_RG32F:
        case GL_RG32I:
        case GL_RG32UI:
        case GL_RG16:
        case GL_RG16F:
        case GL_RG16I:
        case GL_RG16UI:
        case GL_RG8:
        case GL_RG8I:
        case GL_RG8UI:
        case GL_R32F:
        case GL_R32I:
        case GL_R32UI:
        case GL_R16F:
        case GL_R16I:
        case GL_R16UI:
        case GL_R16:
        case GL_R8:
        case GL_R8I:
        case GL_R8UI:
            return GL_TRUE;

        /* Depth/stencil formats - not color renderable. */
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
            return GL_FALSE;

        /* Compressed formats - not color renderable as FBO attachments. */
        case GL_COMPRESSED_RED:
        case GL_COMPRESSED_RG:
        case GL_COMPRESSED_RGB:
        case GL_COMPRESSED_RGBA:
        case GL_COMPRESSED_SRGB:
        case GL_COMPRESSED_SRGB_ALPHA:
        case GL_COMPRESSED_RED_RGTC1:
        case GL_COMPRESSED_SIGNED_RED_RGTC1:
        case GL_COMPRESSED_RG_RGTC2:
        case GL_COMPRESSED_SIGNED_RG_RGTC2:
        case GL_COMPRESSED_RGBA_BPTC_UNORM:
        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
        case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
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
            return GL_FALSE;

        /* Legacy luminance/alpha formats - not color renderable in core profile. */
        case GL_ALPHA:
        case GL_ALPHA8:
        case GL_ALPHA16:
        case GL_ALPHA32F_ARB:
        case GL_ALPHA16F_ARB:
        case GL_LUMINANCE:
        case GL_LUMINANCE8:
        case GL_LUMINANCE16:
        case GL_LUMINANCE32F_ARB:
        case GL_LUMINANCE16F_ARB:
        case GL_LUMINANCE_ALPHA:
        case GL_LUMINANCE_ALPHA32F_ARB:
        case GL_LUMINANCE_ALPHA16F_ARB:
            return GL_FALSE;

        /* SNORM formats - not in the required color-renderable list. */
        case GL_R8_SNORM:
        case GL_RG8_SNORM:
        case GL_RGB8_SNORM:
        case GL_RGBA8_SNORM:
        case GL_R16_SNORM:
        case GL_RG16_SNORM:
        case GL_RGB16_SNORM:
        case GL_RGBA16_SNORM:
            return GL_FALSE;

        /* RGB-only sized formats not in the required list. */
        case GL_RGB8:
        case GL_SRGB8:
        case GL_RGB16:
        case GL_RGB16F:
        case GL_RGB8I:
        case GL_RGB8UI:
        case GL_RGB16I:
        case GL_RGB16UI:
        case GL_RGB10:
        case GL_RGB12:
            return GL_FALSE;

        /* GL_RGB9_E5 is NOT color-renderable per GL 4.6 spec table 8.11,
         * even though Metal supports it as a render target.  Reporting it
         * as renderable caused FBO completeness to pass when the spec
         * requires GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT. */
        case GL_RGB9_E5:
            return GL_FALSE;

        /* RGB32F/I/UI are mapped to RGBA32 Metal formats which are
         * color-renderable.  Treating them as color-renderable avoids
         * InternalError in CTS direct_state_access tests that use these
         * formats for framebuffer attachments. */
        case GL_RGB32F:
        case GL_RGB32I:
        case GL_RGB32UI:
            return GL_TRUE;

        /* Other non-required formats. */
        case GL_R3_G3_B2:
        case GL_RGB4:
        case GL_RGB5:
        case GL_RGBA2:
        case GL_RGBA12:
            return GL_FALSE;

        default:
            return GL_FALSE;
    }
}

uint32_t mtlPixelFormatForGLFormatType(GLenum gl_format, GLenum gl_type)
{
    switch(gl_format)
    {
        case GL_DEPTH_COMPONENT16:
            return MGLPixelFormatDepth16Unorm;

        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
            // Metal has no portable pure D24 texture on Apple Silicon. Use D32F
            // so default GL_DEPTH_COMPONENT24 contexts still get a depth target.
            return MGLPixelFormatDepth32Float;

        case GL_DEPTH_STENCIL:
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
            return MGLPixelFormatDepth32Float_Stencil8;

        case GL_STENCIL_INDEX8:
            return MGLPixelFormatStencil8;

        default:
            break;
    }

    switch(gl_type)
    {
        case GL_UNSIGNED_BYTE:
            switch(gl_format)
            {
                case GL_RED: return MGLPixelFormatR8Uint;
                case GL_RG: return MGLPixelFormatRG8Uint;
                case GL_RGBA: return MGLPixelFormatRGBA8Unorm;
                default:
                    return 0;
            }
            break;

        case GL_BYTE:
            switch(gl_format)
            {
                case GL_RED: return MGLPixelFormatR8Sint;
                case GL_RG: return MGLPixelFormatRG8Sint;
                case GL_RGBA: return MGLPixelFormatRGBA8Sint;
                default:
                    return 0;
            }
            break;

        case GL_UNSIGNED_SHORT:
            switch(gl_format)
            {
                case GL_RED: return MGLPixelFormatR16Uint;
                case GL_RG: return MGLPixelFormatRG16Uint;
                case GL_RGBA: return MGLPixelFormatRGBA16Uint;
                default:
                    return 0;
            }
            break;

        case GL_SHORT:
            switch(gl_format)
            {
                case GL_RED: return MGLPixelFormatR16Sint;
                case GL_RG: return MGLPixelFormatRG16Sint;
                case GL_RGBA: return MGLPixelFormatRGBA16Sint;
                default:
                    return 0;
            }
            break;

        case GL_UNSIGNED_INT:
            switch(gl_format)
            {
                case GL_RED: return MGLPixelFormatR32Uint;
                case GL_RG: return MGLPixelFormatRG32Uint;
                case GL_RGBA: return MGLPixelFormatRGBA32Uint;
                case GL_BGRA: return MGLPixelFormatRGBA32Uint;
                default:
                    return 0;
            }
            break;

        case GL_INT:
            switch(gl_format)
            {
                case GL_RED: return MGLPixelFormatR32Uint;
                case GL_RG: return MGLPixelFormatRG32Uint;
                case GL_RGBA: return MGLPixelFormatRGBA32Uint;
                default:
                    return 0;
            }
            break;

        case GL_FLOAT:
            switch(gl_format)
            {
                case GL_RED: return MGLPixelFormatR32Float;
                case GL_RG: return MGLPixelFormatRG32Float;
                case GL_RGBA: return MGLPixelFormatRGBA32Float;
                case GL_DEPTH_COMPONENT: return MGLPixelFormatDepth32Float;
                case GL_DEPTH_STENCIL: return MGLPixelFormatDepth32Float_Stencil8;

                default:
                    return 0;
            }
            break;

        case GL_HALF_FLOAT:
            switch(gl_format)
            {
                case GL_RED: return MGLPixelFormatR16Float;
                case GL_RG: return MGLPixelFormatRG16Float;
                case GL_RGBA: return MGLPixelFormatRGBA16Float;
                default:
                    return 0;
            }
            break;

        case GL_UNSIGNED_BYTE_3_3_2:
            return 0;

        case GL_UNSIGNED_BYTE_2_3_3_REV:
            return 0;

        case GL_UNSIGNED_SHORT_5_6_5:
            if (__builtin_available(macOS 11.0, *)) {
                return MGLPixelFormatB5G6R5Unorm;
            } else {
                // Fallback on earlier versions
                return MGLPixelFormatInvalid;
            }

        case GL_UNSIGNED_SHORT_5_6_5_REV:
            if (__builtin_available(macOS 11.0, *)) {
                return MGLPixelFormatA1BGR5Unorm;
            } else {
                // Fallback on earlier versions
                return MGLPixelFormatInvalid;
            }

        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
            return 0;

        case GL_UNSIGNED_INT_8_8_8_8:
            return MGLPixelFormatRGBA8Unorm;

        case GL_UNSIGNED_INT_8_8_8_8_REV:
            return MGLPixelFormatBGRA8Unorm;

        case GL_UNSIGNED_INT_10_10_10_2:
            return MGLPixelFormatRGB10A2Unorm;

        case GL_UNSIGNED_INT_2_10_10_10_REV:
            return MGLPixelFormatBGR10A2Unorm;
            return 0;

        default:
            fprintf(stderr,
                    "MGL WARNING: mtlPixelFormatForGLFormatType unknown type 0x%x format 0x%x\n",
                    gl_type,
                    gl_format);
            return MGLPixelFormatInvalid;
    }
}

extern bool mglTexLevelInternalFormatCompressed(GLint internalformat);
extern GLint mglCompressedInternalFormatToSizedUncompressed(GLint internalformat);

uint32_t mtlPixelFormatForGLTex(Texture * tex)
{
    uint32_t mtl_format;
    GLenum internal_format;

    if (!tex)
    {
        fprintf(stderr, "MGL WARNING: mtlPixelFormatForGLTex called with NULL texture\n");
        return MGLPixelFormatInvalid;
    }

    internal_format = tex->internalformat;
    if (!internal_format)
    {
        fprintf(stderr,
                "MGL WARNING: mtlPixelFormatForGLTex texture %u has no internal format\n",
                tex->name);
        return MGLPixelFormatInvalid;
    }

    /* glTexImage* with a compressed internalformat stores data uncompressed
     * (see mglCompressedInternalFormatToSizedUncompressed remap in
     * createTextureLevel). The Metal texture must match the uncompressed
     * data layout, not the compressed internalformat. glCompressedTexImage*
     * stores data with pitch==0 and still needs the compressed Metal format. */
    if (mglTexLevelInternalFormatCompressed((GLint)internal_format) &&
        tex->faces[0].levels &&
        tex->faces[0].levels[0].pitch > 0u)
    {
        internal_format = (GLenum)mglCompressedInternalFormatToSizedUncompressed((GLint)internal_format);
    }

    mtl_format = mtlFormatForGLInternalFormat(internal_format);
    if (mtl_format == MGLPixelFormatInvalid)
    {
        fprintf(stderr,
                "MGL WARNING: mtlPixelFormatForGLTex texture %u unsupported internal format 0x%x\n",
                tex->name,
                internal_format);
        return MGLPixelFormatInvalid;
    }

    if (tex->is_render_target &&
        mtl_format == MGLPixelFormatDepth32Float) {
        mtl_format = MGLPixelFormatDepth32Float_Stencil8;
    }

    return mtl_format;
}

#include <math.h>
#include <string.h>

float mglHalfToFloat(uint16_t value)
{
    uint32_t sign = (uint32_t)(value >> 15u);
    uint32_t exponent = (value >> 10u) & 31u;
    uint32_t mantissa = value & 1023u;
    float result;
    if (exponent == 0u) {
        result = ldexpf((float)mantissa, -24);
    } else if (exponent == 31u) {
        result = mantissa ? NAN : INFINITY;
    } else {
        result = ldexpf(1.0f + (float)mantissa / 1024.0f, (int)exponent - 15);
    }
    return sign ? -result : result;
}

uint16_t mglFloatToHalf(float value)
{
    uint32_t f;
    memcpy(&f, &value, sizeof(f));
    uint32_t sign = (f >> 16u) & 0x8000u;
    int32_t exp = ((int32_t)(f >> 23u) & 0xff) - 112;
    uint32_t mant = f & 0x7fffffu;

    /* Handle NaN and Inf: IEEE-754 float exp field == 0xFF (exp == 143). */
    if (exp >= 143) {
        if (mant != 0u) {
            /* NaN — preserve sign and set half-float NaN (exp=31, mant!=0). */
            return (uint16_t)(sign | 0x7e00u);
        }
        /* Infinity. */
        return (uint16_t)(sign | 0x7c00u);
    }

    if (exp <= 0) {
        /* Denormalized or zero in half-float.
         * Guard against undefined shift (1 - exp can exceed 32 for very
         * small values).  Values smaller than 2^-24 round to zero. */
        int shift = 1 - exp;
        if (shift >= 25) {
            return (uint16_t)sign;  /* ±0 */
        }
        /* Round-to-nearest-even before truncation. */
        uint32_t m = (mant | 0x800000u) >> shift;
        /* Add rounding bias (bit 12 = 1 << (13-1)) and round-to-even. */
        m += 0x00001000u + ((m >> 13u) & 1u);
        return (uint16_t)(sign | (m >> 13u));
    }
    if (exp >= 31) {
        /* Overflow to Infinity. */
        return (uint16_t)(sign | 0x7c00u);
    }
    /* Normalized: round-to-nearest-even. */
    mant += 0x00001000u + ((mant >> 13u) & 1u);
    if (mant >= 0x800000u) {
        mant = 0;
        exp++;
        if (exp >= 31) {
            return (uint16_t)(sign | 0x7c00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10u) | (mant >> 13u));
}

/* Pack a float into 11-bit unsigned float (UE11) format.
 * 6-bit mantissa, 5-bit exponent (bias 15).
 * Special values: exp=31,mant=0 → Infinity; exp=31,mant!=0 → NaN.
 * Uses round-to-nearest-even when truncating the 23-bit IEEE mantissa to
 * 6 bits, per the GL_UNSIGNED_INT_10F_11F_11F_REV packing spec. */
uint32_t mglFloatToFloat11(float v)
{
    if (isnan(v)) return 0x7e0u;  /* NaN: exp=31, mant!=0 */
    if (v <= 0.0f) return 0u;
    if (v >= 65024.0f) return 0x7c0u; /* Infinity: exp=31, mant=0 */
    /* Convert to IEEE-754 half float first, then extract mantissa/exponent */
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    int ieee_exp = (int)((bits >> 23) & 0xff) - 127;
    uint32_t ieee_mant = bits & 0x7fffff;
    if (ieee_exp <= -15) {
        /* Denormalized in float11 */
        int shift = -14 - ieee_exp;
        if (shift >= 11) return 0u;
        /* Round-to-nearest-even for denormals. */
        uint32_t src = (ieee_mant | 0x800000);
        int rshift = 23 - 6 + shift;
        uint32_t m = src >> rshift;
        uint32_t rem = src & ((1u << rshift) - 1u);
        uint32_t half = 1u << (rshift - 1);
        if (rem > half || (rem == half && (m & 1u))) {
            m += 1u;
        }
        return m & 0x3fu;
    }
    if (ieee_exp >= 16) return 0x7c0u; /* Infinity */
    uint32_t exp = (uint32_t)(ieee_exp + 15);
    /* Round-to-nearest-even: 23-bit mantissa → 6-bit. */
    uint32_t mant = ieee_mant >> (23 - 6);
    uint32_t rem = ieee_mant & ((1u << (23 - 6)) - 1u);
    uint32_t half = 1u << (23 - 6 - 1);
    if (rem > half || (rem == half && (mant & 1u))) {
        mant += 1u;
        /* Carry into exponent if mantissa overflows. */
        if (mant > 0x3fu) {
            mant = 0u;
            exp += 1u;
            if (exp >= 31u) return 0x7c0u; /* overflow to Infinity */
        }
    }
    return (exp << 6) | mant;
}

/* Pack a float into 10-bit unsigned float (UE10) format.
 * 5-bit mantissa, 5-bit exponent.
 * Special values: exp=31,mant=0 → Infinity; exp=31,mant!=0 → NaN.
 * Uses round-to-nearest-even when truncating the 23-bit IEEE mantissa to
 * 5 bits. */
uint32_t mglFloatToFloat10(float v)
{
    if (isnan(v)) return 0x3f0u;  /* NaN: exp=31, mant!=0 */
    if (v <= 0.0f) return 0u;
    if (v >= 64512.0f) return 0x3e0u; /* Infinity: exp=31, mant=0 */
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    int ieee_exp = (int)((bits >> 23) & 0xff) - 127;
    uint32_t ieee_mant = bits & 0x7fffff;
    if (ieee_exp <= -15) {
        int shift = -14 - ieee_exp;
        if (shift >= 10) return 0u;
        /* Round-to-nearest-even for denormals. */
        uint32_t src = (ieee_mant | 0x800000);
        int rshift = 23 - 5 + shift;
        uint32_t m = src >> rshift;
        uint32_t rem = src & ((1u << rshift) - 1u);
        uint32_t half = 1u << (rshift - 1);
        if (rem > half || (rem == half && (m & 1u))) {
            m += 1u;
        }
        return m & 0x1fu;
    }
    if (ieee_exp >= 16) return 0x3e0u; /* Infinity */
    uint32_t exp = (uint32_t)(ieee_exp + 15);
    /* Round-to-nearest-even: 23-bit mantissa → 5-bit. */
    uint32_t mant = ieee_mant >> (23 - 5);
    uint32_t rem = ieee_mant & ((1u << (23 - 5)) - 1u);
    uint32_t half = 1u << (23 - 5 - 1);
    if (rem > half || (rem == half && (mant & 1u))) {
        mant += 1u;
        if (mant > 0x1fu) {
            mant = 0u;
            exp += 1u;
            if (exp >= 31u) return 0x3e0u;
        }
    }
    return (exp << 5) | mant;
}

/* Pack 3 RGB floats into GL_UNSIGNED_INT_5_9_9_9_REV (GL_RGB9_E5) format.
 * All 3 mantissas share one 5-bit exponent. Implements the shared exponent
 * algorithm from the GL spec (and matches CTS glcPackedPixelsTests
 * unpack_UNSIGNED_INT_5_9_9_9_REV). */
uint32_t mglPackRGBToSharedExp(double red, double green, double blue)
{
    const int N     = 9;   /* mantissa bits */
    const int B     = 15;  /* exponent bias */
    const int E_max = 31;  /* max exponent */

    /* sharedExpMax = (2^N - 1) / 2^N * 2^(E_max - B) */
    double shared_exp_max = ((double)((1 << N) - 1) / (double)(1 << N)) *
                            ldexp(1.0, E_max - B);

    double red_c   = fmax(0.0, fmin(shared_exp_max, red));
    double green_c = fmax(0.0, fmin(shared_exp_max, green));
    double blue_c  = fmax(0.0, fmin(shared_exp_max, blue));

    double max_c = fmax(fmax(red_c, green_c), blue_c);

    double exp_p;
    if (max_c <= 0.0) {
        exp_p = 0.0;
    } else {
        /* CTS formula: exp_p = max(-B-1, floor(log2(max_c))) + 1 + B */
        exp_p = fmax((double)(-B - 1), floor(log2(max_c))) + 1.0 + (double)B;
    }

    /* Check if max_s overflows; if so, increment exp_s.
     * CTS: max_s = floor(max_c / 2^(exp_p - B - N) + 0.5)
     * if max_s >= 2^N, exp_s = exp_p + 1, else exp_s = exp_p */
    double scale_p = ldexp(1.0, (int)exp_p - B - N);
    double max_s = floor(max_c / scale_p + 0.5);

    int exp_s;
    if (max_s >= (double)(1 << N)) {
        exp_s = (int)exp_p + 1;
    } else {
        exp_s = (int)exp_p;
    }
    if (exp_s < 0) exp_s = 0;
    if (exp_s > E_max) exp_s = E_max;

    /* scale = 2^(exp_s - B - N) per CTS */
    double scale = ldexp(1.0, exp_s - B - N);

    uint32_t red_s   = (uint32_t)floor(red_c   / scale + 0.5);
    uint32_t green_s = (uint32_t)floor(green_c / scale + 0.5);
    uint32_t blue_s  = (uint32_t)floor(blue_c  / scale + 0.5);

    if (red_s > 511u) red_s = 511u;
    if (green_s > 511u) green_s = 511u;
    if (blue_s > 511u) blue_s = 511u;

    return red_s | (green_s << 9) | (blue_s << 18) | ((uint32_t)exp_s << 27);
}

/* Unpack a GL_UNSIGNED_INT_5_9_9_9_REV (GL_RGB9_E5) packed value to 3 doubles.
 * Layout: R[0:8], G[9:17], B[18:26], shared_exp[27:31].
 * CTS: value = mantissa * 2^(exponent - B - N)  where N=9, B=15. */
void mglUnpackSharedExp(uint32_t packed, double *r, double *g, double *b)
{
    const int N = 9;
    const int B = 15;
    uint32_t mant_r = packed & 511u;
    uint32_t mant_g = (packed >> 9u) & 511u;
    uint32_t mant_b = (packed >> 18u) & 511u;
    uint32_t exp = (packed >> 27u) & 31u;

    double scale = ldexp(1.0, (int)exp - B - N);
    *r = (double)mant_r * scale;
    *g = (double)mant_g * scale;
    *b = (double)mant_b * scale;
}

uint32_t mglPackUnsignedFloatFromUNorm8(uint32_t value, uint32_t mantissa_bits)
{
    if (value == 0u || mantissa_bits == 0u || mantissa_bits > 23u) {
        return 0u;
    }

    float scaled = (float)value / 255.0f;
    int exponent = 15;
    while (scaled < 1.0f && exponent > 0) {
        scaled *= 2.0f;
        exponent--;
    }
    while (scaled >= 2.0f && exponent < 31) {
        scaled *= 0.5f;
        exponent++;
    }

    uint32_t mantissa_mask = (1u << mantissa_bits) - 1u;
    uint32_t mantissa = 0u;
    if (exponent == 0) {
        float subnormal = (float)value / 255.0f;
        for (uint32_t i = 0; i < mantissa_bits + 14u; i++) {
            subnormal *= 2.0f;
        }
        mantissa = (uint32_t)(subnormal + 0.5f);
        if (mantissa > mantissa_mask) {
            mantissa = mantissa_mask;
        }
    } else {
        float frac = (scaled - 1.0f) * (float)(1u << mantissa_bits);
        mantissa = (uint32_t)(frac + 0.5f);
        if (mantissa > mantissa_mask) {
            mantissa = 0u;
            if (exponent < 31) {
                exponent++;
            } else {
                mantissa = mantissa_mask;
            }
        }
    }

    return ((uint32_t)exponent << mantissa_bits) | (mantissa & mantissa_mask);
}

float mglUnpackUnsignedFloatComponent(uint32_t value, uint32_t mantissa_bits)
{
    if (mantissa_bits == 0u || mantissa_bits > 23u) return 0.0f;

    uint32_t mantissa_mask = (1u << mantissa_bits) - 1u;
    uint32_t mantissa = value & mantissa_mask;
    uint32_t exponent = (value >> mantissa_bits) & 0x1fu;

    if (exponent == 31u) {
        /* Inf or NaN. */
        return (mantissa == 0u) ? INFINITY : NAN;
    }

    /* value = (1 + mantissa / 2^mantissa_bits) * 2^(exponent - 15)
     * For exponent == 0 (subnormal): value = mantissa / 2^mantissa_bits * 2^(1-15) */
    if (exponent == 0u) {
        return ldexpf((float)mantissa, 1 - 15 - (int)mantissa_bits);
    }
    float normalized = 1.0f + (float)mantissa / (float)(1u << mantissa_bits);
    return ldexpf(normalized, (int)exponent - 15);
}
