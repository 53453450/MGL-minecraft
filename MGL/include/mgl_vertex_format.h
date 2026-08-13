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

#include <objc/objc.h>  /* BOOL, GLuint, GLenum, GLboolean */
#include <stdint.h>
#include <stddef.h>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

#include "glm_context.h"
#include "mgl_byte_hash.h"

#ifdef __cplusplus
extern "C" {
#endif

/* === Vertex format mapping (static inline, hot-path) === */

/* GL vertex attribute component size in bytes (1/2/4/8).  Returns 0 for
 * unknown types. */
static inline size_t mglVertexAttribComponentSize(GLenum type)
{
    switch (type)
    {
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

/* GL index element size in bytes (1/2/4).  Returns 0 for unknown types. */
static inline NSUInteger mglGLIndexElementSize(GLenum type)
{
    switch (type) {
        case GL_UNSIGNED_BYTE: return 1u;
        case GL_UNSIGNED_SHORT: return 2u;
        case GL_UNSIGNED_INT: return 4u;
        default: return 0u;
    }
}

/* Reads a single GL index value from a byte buffer.  Returns 0 for NULL
 * buffer or unknown type. */
static inline uint32_t mglReadGLIndexValue(const uint8_t *indexBytes,
                                           GLenum type,
                                           NSUInteger elementIndex)
{
    if (!indexBytes) {
        return 0u;
    }

    switch (type) {
        case GL_UNSIGNED_BYTE: {
            uint8_t v = 0;
            memcpy(&v, indexBytes + elementIndex, sizeof(v));
            return (uint32_t)v;
        }
        case GL_UNSIGNED_SHORT: {
            uint16_t v = 0;
            memcpy(&v, indexBytes + (elementIndex * 2u), sizeof(v));
            return (uint32_t)v;
        }
        case GL_UNSIGNED_INT: {
            uint32_t v = 0;
            memcpy(&v, indexBytes + (elementIndex * 4u), sizeof(v));
            return v;
        }
        default:
            return 0u;
    }
}

/* Total bytes for a vertex attrib element (type × size), with special
 * handling for packed 10_10_10_2 formats.  Returns 0 for unknown. */
static inline size_t mglVertexAttribElementBytes(GLenum type, GLuint size)
{
    switch (type) {
        case GL_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
            return 4u;
        default: {
            size_t comp = mglVertexAttribComponentSize(type);
            if (comp == 0u || size == 0u) {
                return 0u;
            }
            return comp * (size_t)size;
        }
    }
}

/* Maps a GL double attrib size to the corresponding MTLVertexFormat (Float
 * variants, since Metal has no double vertex formats). */
static inline MTLVertexFormat mglDoubleVertexAttribFloatFormat(GLuint size)
{
    switch (size) {
        case 1: return MTLVertexFormatFloat;
        case 2: return MTLVertexFormatFloat2;
        case 3: return MTLVertexFormatFloat3;
        case 4: return MTLVertexFormatFloat4;
        default: return MTLVertexFormatInvalid;
    }
}

/* Aligns a vertex stride to Metal's 4-byte minimum alignment. */
static inline NSUInteger mglAlignVertexStrideForMetal(NSUInteger stride)
{
    return (stride + 3u) & ~(NSUInteger)3u;
}

/* === Vertex format mapping (extern) === */

/* Human-readable name for an MTLVertexFormat enum value. */
const char *mglVertexFormatName(MTLVertexFormat format);

/* For glVertexAttribIFormat (integer) attribs, Metal only allows 32-bit
 * Int/UInt formats.  Returns true and sets *outFormat when CPU conversion
 * is required; false otherwise. */
bool mglIntegerAttribNeedsConversion(GLenum srcType,
                                     GLuint shaderGlType,
                                     GLuint size,
                                     MTLVertexFormat *outFormat);

/* Decodes a single vertex attrib component to double for trace/replay.
 * Handles normalized conversion for signed/unsigned types. */
double mglDecodeVertexAttribComponent(const uint8_t *src,
                                      GLenum type,
                                      GLboolean normalized,
                                      NSUInteger component);

/* === Pipeline signature === */

/* FNV-1a hash of a MTLVertexDescriptor for pipeline cache keys. */
uint64_t mglVertexDescriptorSignature(MTLVertexDescriptor *vertexDescriptor);

/* FNV-1a hash of a MTLRenderPipelineDescriptor for pipeline cache keys. */
uint64_t mglPipelineDescriptorSignature(MTLRenderPipelineDescriptor *pipelineStateDescriptor);

/* P4.2: value-state 版签名（MGLRenderCppPipelineDescriptorState 的完整定义
 * 在 mgl_air_loader.h）。哈希字段与顺序必须与 descriptor 版完全一致，保证
 * Metal-cpp 路径的 pipeline cache key 语义不变。 */
typedef struct MGLRenderCppPipelineDescriptorState
    MGLRenderCppPipelineDescriptorState;
uint64_t mglVertexDescriptorSignatureFromState(
    const MGLRenderCppPipelineDescriptorState *state);
uint64_t mglPipelineDescriptorSignatureFromState(
    const MGLRenderCppPipelineDescriptorState *state);

/* Inverts MTLWinding (CW↔CCW) when `invert` is true; otherwise returns
 * winding unchanged. */
MTLWinding mglMaybeInvertMTLWinding(MTLWinding winding, BOOL invert);

#ifdef __cplusplus
}
#endif

#endif /* MGL_VERTEX_FORMAT_H */
