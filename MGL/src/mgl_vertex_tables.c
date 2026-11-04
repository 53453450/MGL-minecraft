/* SPDX-License-Identifier: LGPL-3.0-only */
/*
 * mgl_vertex_tables.c
 * MGL
 *
 * GL vertex-attribute tables: bytes per component of a GL type, and the
 * conversion descriptor for a shader attribute kind.  Pure tables; the
 * declarations live in mgl_vertex_format.h / mgl_render_api_binding.h.
 */

#include "mgl_vertex_format.h"
#include "mgl_render_api_binding.h"
#include "mgl_render_api_buffer.h"

#include <string.h>

uint32_t mglRenderVertexAttribComponentSize(uint64_t gl_type) {
    switch (gl_type) {
        case GL_BYTE:
        case GL_UNSIGNED_BYTE:
            return 1u;
        case GL_SHORT:
        case GL_UNSIGNED_SHORT:
        case GL_HALF_FLOAT:
            return 2u;
        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_FLOAT:
        case GL_FIXED:
        case GL_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
            return 4u;
        case GL_DOUBLE:
            return 8u;
        default:
            return 0u;
    }
}

int mglRenderFillVertexConversionFromAttribKind(
    int attrib_kind, uint32_t size, uint32_t type, int normalized,
    int dst_signed, MGLRenderVertexConversion *out) {
    if (!out) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    switch (attrib_kind) {
        case MGL_ATTRIB_CONV_DOUBLE:
            if (size == 0u || size > 4u) {
                return -1;
            }
            out->kind = MGL_RENDER_VERTEX_DOUBLE_TO_FLOAT;
            out->component_count = size;
            out->source_type = GL_DOUBLE;
            break;
        case MGL_ATTRIB_CONV_INT_TO_FLOAT:
            if (size == 0u || size > 4u ||
                (type != GL_INT && type != GL_UNSIGNED_INT)) {
                return -1;
            }
            out->kind = MGL_RENDER_VERTEX_INT_TO_FLOAT;
            out->component_count = size;
            out->source_type = type;
            out->normalized = normalized ? 1u : 0u;
            break;
        case MGL_ATTRIB_CONV_FIXED:
            if (size == 0u || size > 4u) {
                return -1;
            }
            out->kind = MGL_RENDER_VERTEX_FIXED_TO_FLOAT;
            out->component_count = size;
            out->source_type = GL_FIXED;
            break;
        case MGL_ATTRIB_CONV_UINT_1010102:
            out->kind = MGL_RENDER_VERTEX_PACKED_1010102_TO_FLOAT;
            out->component_count = 4u;
            out->source_type = GL_UNSIGNED_INT_10_10_10_2;
            out->normalized = 1u;
            break;
        case MGL_ATTRIB_CONV_REV2101010:
            if (type != GL_UNSIGNED_INT_2_10_10_10_REV &&
                type != GL_INT_2_10_10_10_REV) {
                return -1;
            }
            out->kind = MGL_RENDER_VERTEX_PACKED_1010102_TO_FLOAT;
            out->component_count = 4u;
            out->source_type = type;
            out->normalized = normalized ? 1u : 0u;
            out->destination_signed = (size == GL_BGRA) ? 1u : 0u;
            break;
        case MGL_ATTRIB_CONV_UINT_10F11F11F:
            out->kind = MGL_RENDER_VERTEX_PACKED_10F11F11F_TO_FLOAT;
            out->component_count = 3u;
            out->source_type = GL_UNSIGNED_INT_10F_11F_11F_REV;
            break;
        case MGL_ATTRIB_CONV_INTEGER_SIGN:
            if (size == 0u || size > 4u) {
                return -1;
            }
            out->kind = MGL_RENDER_VERTEX_INTEGER_TO_32;
            out->component_count = size;
            out->source_type = type;
            out->destination_signed = dst_signed ? 1u : 0u;
            break;
        default:
            return -1;
    }
    return 0;
}
