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
 * mgl_vertex_format.h
 * MGL
 *
 * Vertex Format / Pipeline Signature Subsystem: pure helpers for translating
 * OpenGL vertex attribute and index formats to Metal, plus pipeline/vertex
 * descriptor signature computation for pipeline state caching.
 *
 * Two groups:
 *
 *   1. Vertex format mapping (9 functions): GL component size, MTLVertexFormat
 *     naming, GL index element size/value, vertex attrib element bytes, double
 *     format expansion, integer attrib conversion check, stride alignment,
 *     and component decoding for trace/replay.
 *
 *   2. Pipeline signature (3 functions): FNV-1a hashing of
 *     MTLVertexDescriptor and MTLRenderPipelineDescriptor for cache keys,
 *     plus MTLWinding inversion helper.
 *
 * All functions are pure: no self/ivar dependency.  They operate on Metal
 * descriptor objects and GL enum values.
 *
 * Dependencies:
 *   - glm_context.h (GL enums, MAX_ATTRIBS, MAX_COLOR_ATTACHMENTS)
 *   - mgl_byte_hash.h (mglHashStepU64, static inline)
 *   - Metal.framework (MTLVertexFormat, MTLVertexDescriptor, etc.)
 *
 * NOTE: kMGLMaxMetalVertexBufferCount (31) is used in the pipeline signature
 * implementation.  It is intentionally referenced as a literal in the .m file
 * to avoid a circular include with MGLRenderer.m, which defines its own
 * static const version of the same constant.
 */

#ifndef MGL_VERTEX_FORMAT_H
#define MGL_VERTEX_FORMAT_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stddef.h>

#include "glm_context.h"
#include "mgl_byte_hash.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward decls: the GL index element-size / read-index-value logic lives in
 * mgl_render.cpp (both gates); the inlines below delegate to it. */
uint32_t mglRenderGLIndexElementSize(uint64_t gl_index_type);
uint32_t mglRenderReadGLIndexValue(const uint8_t *bytes, uint32_t elem_width,
                                      uint64_t element_index);
/* Vertex format class: kind * 4 + component count, where kind is 1 = float,
 * 2 = int, 3 = uint; 0 means "not a vertex scalar/vector type".  The class is
 * GL-free so the backend can map it to MTL::VertexFormat without reading GL
 * enums (C2).  Inline: every user needs only a handful of cases, and the
 * Metal-cpp smoke test links a subset of the objects. */
static inline uint32_t mglRenderVertexFormatClass(uint64_t gl_type)
{
    switch (gl_type) {
        case GL_FLOAT: return 1u * 4u + 1u;
        case GL_FLOAT_VEC2: return 1u * 4u + 2u;
        case GL_FLOAT_VEC3: return 1u * 4u + 3u;
        case GL_FLOAT_VEC4: return 1u * 4u + 4u;
        case GL_INT: return 2u * 4u + 1u;
        case GL_INT_VEC2: return 2u * 4u + 2u;
        case GL_INT_VEC3: return 2u * 4u + 3u;
        case GL_INT_VEC4: return 2u * 4u + 4u;
        case GL_UNSIGNED_INT:
        case GL_BOOL: return 3u * 4u + 1u;
        case GL_UNSIGNED_INT_VEC2:
        case GL_BOOL_VEC2: return 3u * 4u + 2u;
        case GL_UNSIGNED_INT_VEC3:
        case GL_BOOL_VEC3: return 3u * 4u + 3u;
        case GL_UNSIGNED_INT_VEC4:
        case GL_BOOL_VEC4: return 3u * 4u + 4u;
        default: return 0u;
    }
}

/* Same question for one 16-byte location of a control-point input: a matrix
 * type occupies one location per column, so its class is that of the column
 * vector. */
static inline uint32_t mglRenderVertexLocationFormatClass(uint64_t gl_type)
{
    switch (gl_type) {
        case GL_FLOAT_MAT2:
        case GL_FLOAT_MAT3x2:
        case GL_FLOAT_MAT4x2:
            return 1u * 4u + 2u;
        case GL_FLOAT_MAT3:
        case GL_FLOAT_MAT2x3:
        case GL_FLOAT_MAT4x3:
            return 1u * 4u + 3u;
        case GL_FLOAT_MAT4:
        case GL_FLOAT_MAT2x4:
        case GL_FLOAT_MAT3x4:
            return 1u * 4u + 4u;
        default:
            return mglRenderVertexFormatClass(gl_type);
    }
}

/* GL draw mode -> primitive class (0 point, 1 line, 2 line strip, 3 triangle,
 * 4 triangle strip), 0xFFFFFFFFu when the mode routes elsewhere. */
static inline uint32_t mglRenderPrimitiveClassForGLMode(uint32_t mode)
{
    switch (mode) {
        case GL_POINTS: return 0u;
        case GL_LINES: return 1u;
        case GL_LINE_STRIP: return 2u;
        case GL_TRIANGLES: return 3u;
        case GL_TRIANGLE_STRIP: return 4u;
        default: return 0xFFFFFFFFu;
    }
}

/* How the client stores one attribute component.  The backend asks this
 * instead of comparing GL enums, so the GL vocabulary stays in the frontend. */
typedef enum MGLVertexFetchKind {
    MGL_VFETCH_KIND_INVALID = 0,
    MGL_VFETCH_KIND_S8,
    MGL_VFETCH_KIND_U8,
    MGL_VFETCH_KIND_S16,
    MGL_VFETCH_KIND_U16,
    MGL_VFETCH_KIND_S32,
    MGL_VFETCH_KIND_U32,
    MGL_VFETCH_KIND_F16,
    MGL_VFETCH_KIND_F32,
    MGL_VFETCH_KIND_F64,
    MGL_VFETCH_KIND_FIXED,
    MGL_VFETCH_KIND_PACKED_10_10_10_2,
    MGL_VFETCH_KIND_PACKED_10F_11F_11F,
    MGL_VFETCH_KIND_PACKED_2_10_10_10_REV,
    MGL_VFETCH_KIND_PACKED_2_10_10_10_REV_INT
} MGLVertexFetchKind;

static inline MGLVertexFetchKind mglVertexFetchKindForGLType(uint32_t type)
{
    switch (type) {
        case GL_BYTE: return MGL_VFETCH_KIND_S8;
        case GL_UNSIGNED_BYTE: return MGL_VFETCH_KIND_U8;
        case GL_SHORT: return MGL_VFETCH_KIND_S16;
        case GL_UNSIGNED_SHORT: return MGL_VFETCH_KIND_U16;
        case GL_INT: return MGL_VFETCH_KIND_S32;
        case GL_UNSIGNED_INT: return MGL_VFETCH_KIND_U32;
        case GL_HALF_FLOAT: return MGL_VFETCH_KIND_F16;
        case GL_FLOAT: return MGL_VFETCH_KIND_F32;
        case GL_DOUBLE: return MGL_VFETCH_KIND_F64;
        case GL_FIXED: return MGL_VFETCH_KIND_FIXED;
        case GL_UNSIGNED_INT_10_10_10_2:
            return MGL_VFETCH_KIND_PACKED_10_10_10_2;
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
            return MGL_VFETCH_KIND_PACKED_10F_11F_11F;
        case GL_UNSIGNED_INT_2_10_10_10_REV:
            return MGL_VFETCH_KIND_PACKED_2_10_10_10_REV;
        case GL_INT_2_10_10_10_REV:
            return MGL_VFETCH_KIND_PACKED_2_10_10_10_REV_INT;
        default: return MGL_VFETCH_KIND_INVALID;
    }
}

/* The "BGRA" pseudo-size GL passes to glVertexAttribPointer for packed color
 * data.  Kept here so the backend can test it without naming the enum. */
static inline bool mglVertexAttribSizeIsBGRA(uint32_t size)
{
    return size == GL_BGRA;
}

/* Swizzle suffix that selects the components a GLSL type actually uses:
 * ".x", ".xy", ".xyz", or "" when the type already fills the vector.  NULL for
 * types that are not scalar/vector numbers. */
static inline const char *mglGlslTypeSwizzleSuffix(uint32_t type)
{
    switch (type) {
        case GL_FLOAT:
        case GL_INT:
        case GL_UNSIGNED_INT:
            return ".x";
        case GL_FLOAT_VEC2:
        case GL_INT_VEC2:
        case GL_UNSIGNED_INT_VEC2:
            return ".xy";
        case GL_FLOAT_VEC3:
        case GL_INT_VEC3:
        case GL_UNSIGNED_INT_VEC3:
            return ".xyz";
        case GL_FLOAT_VEC4:
        case GL_INT_VEC4:
        case GL_UNSIGNED_INT_VEC4:
            return "";
        default:
            return NULL;
    }
}

/* The float-shaped name of an integer GLSL type ("float"/"vec2"/"vec3"/"vec4"),
 * for code that reads an integer attribute as floats.  NULL for non-integer
 * types. */
static inline const char *mglGlslIntegerAsFloatTypeName(uint32_t type)
{
    switch (type) {
        case GL_INT:
        case GL_UNSIGNED_INT:
            return "float";
        case GL_INT_VEC2:
        case GL_UNSIGNED_INT_VEC2:
            return "vec2";
        case GL_INT_VEC3:
        case GL_UNSIGNED_INT_VEC3:
            return "vec3";
        case GL_INT_VEC4:
        case GL_UNSIGNED_INT_VEC4:
            return "vec4";
        default:
            return NULL;
    }
}

/* GL index type -> element width (2 or 4), 0 when unsupported. */
static inline uint32_t mglRenderIndexElementClass(uint32_t gl_type)
{
    switch (gl_type) {
        case GL_UNSIGNED_BYTE:
        case GL_UNSIGNED_SHORT: return 2u;
        case GL_UNSIGNED_INT: return 4u;
        default: return 0u;
    }
}
uint32_t mglRenderVertexAttribComponentSize(uint64_t gl_type);
uint64_t mglRenderVertexAttribElementBytes(uint64_t gl_type, uint32_t size);
uint64_t mglRenderAlignVertexStrideForMetal(uint64_t stride);
uint32_t mglRenderDoubleVertexAttribFloatFormat(uint32_t size);
uint32_t mglRenderIntegerAttribConversionFormat(
    uint64_t src_type, uint64_t shader_gl_type, uint32_t size);
const char *mglRenderVertexFormatName(uint32_t format);
uint64_t mglRenderVertexDescriptorSignature(const void *descriptor);
uint64_t mglRenderPipelineDescriptorSignature(const void *descriptor);

/* === Vertex format mapping (static inline, hot-path) === */

/* GL vertex attribute component size in bytes (1/2/4/8).  Returns 0 for
 * unknown types.  Pure logic lives in mgl_render.cpp. */
static inline size_t mglVertexAttribComponentSize(GLenum type)
{
    return (size_t)mglRenderVertexAttribComponentSize((uint64_t)type);
}

/* GL index element size in bytes (1/2/4).  Returns 0 for unknown types. */
static inline size_t mglGLIndexElementSize(GLenum type)
{
    return (size_t)mglRenderGLIndexElementSize((uint64_t)type);
}

/* Reads a single GL index value from a byte buffer.  Returns 0 for NULL
 * buffer or unknown type.  Pure logic lives in mgl_render.cpp. */
static inline uint32_t mglReadGLIndexValue(const uint8_t *indexBytes,
                                           GLenum type,
                                           size_t elementIndex)
{
    return mglRenderReadGLIndexValue(
        indexBytes, (uint32_t)mglGLIndexElementSize(type), (uint64_t)elementIndex);
}

/* Total bytes for a vertex attrib element (type × size), with special
 * handling for packed 10_10_10_2 formats.  Returns 0 for unknown.
 * Pure logic lives in mgl_render.cpp. */
static inline size_t mglVertexAttribElementBytes(GLenum type, GLuint size)
{
    return (size_t)mglRenderVertexAttribElementBytes((uint64_t)type, (uint32_t)size);
}

/* Maps a GL double attrib size to the corresponding MTLVertexFormat (Float
 * variants, since Metal has no double vertex formats). */
static inline uint32_t mglDoubleVertexAttribFloatFormat(GLuint size)
{
    return mglRenderDoubleVertexAttribFloatFormat((uint32_t)size);
}

/* Aligns a vertex stride to Metal's 4-byte minimum alignment. */
static inline size_t mglAlignVertexStrideForMetal(size_t stride)
{
    return (size_t)mglRenderAlignVertexStrideForMetal((uint64_t)stride);
}

/* === Vertex format mapping (extern) === */

/* Human-readable name for an MTLVertexFormat enum value. */
const char *mglVertexFormatName(uint32_t format);

/* For glVertexAttribIFormat (integer) attribs, Metal only allows 32-bit
 * Int/UInt formats.  Returns true and sets *outFormat when CPU conversion
 * is required; false otherwise. */
bool mglIntegerAttribNeedsConversion(GLenum srcType,
                                     GLuint shaderGlType,
                                     GLuint size,
                                     void *outFormat);

/* Decodes a single vertex attrib component to double for trace/replay.
 * Handles normalized conversion for signed/unsigned types. */
double mglDecodeVertexAttribComponent(const uint8_t *src,
                                      GLenum type,
                                      GLboolean normalized,
                                      size_t component);

/* === Pipeline signature === */

/* FNV-1a hash of a MTLVertexDescriptor for pipeline cache keys. */
uint64_t mglVertexDescriptorSignature(const void *vertexDescriptor);

/* FNV-1a hash of a MTLRenderPipelineDescriptor for pipeline cache keys. */
uint64_t mglPipelineDescriptorSignature(const void *pipelineStateDescriptor);

/* Value-state signature. Field order must remain stable because it defines the
 * pipeline-cache key. The complete state is defined in mgl_air_loader.h. */
typedef struct MGLRenderPipelineDescriptorState
    MGLRenderPipelineDescriptorState;
uint64_t mglVertexDescriptorSignatureFromState(
    const MGLRenderPipelineDescriptorState *state);
uint64_t mglPipelineDescriptorSignatureFromState(
    const MGLRenderPipelineDescriptorState *state);

/* Inverts MTLWinding (CW↔CCW) when `invert` is true; otherwise returns
 * winding unchanged. */
uint32_t mglMaybeInvertMTLWinding(uint32_t winding, bool invert);

#ifdef __cplusplus
}
#endif

#endif /* MGL_VERTEX_FORMAT_H */
