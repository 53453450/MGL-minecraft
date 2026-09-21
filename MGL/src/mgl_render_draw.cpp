/* SPDX-License-Identifier: LGPL-3.0-only */
#include "mgl_metal.h"
#include "mgl_render.h"
#include "mgl_render_pixel.h"
#include "mgl_index_buffer.h"
extern "C" {
#include "pixel_utils.h"
}
#include "mgl_program_resource.h"
#include "mgl_renderer_backend.h"
#include "mgl_air_loader.h"
#include "mgl_air_tess_abi.h"
#include "mgl_aux_assets.h"
#include "mgl_compute_pipeline_cache.h"
#include "mgl_env_flag.h"
#include "mgl_program_reflection.h"
#include "mgl_types_buffer.h"
#include "mgl_types_texture.h"
#include "mgl_types_program.h"
#include "mgl_types_state.h"
#include "mgl_types_sync.h"
#include "glm_context.h"
#include "mgl_render_internal.h"
#include "mgl_capability.h"
#include "mgl_sync.h"
#include "glm_limits.h"
#include "mgl_shader_abi.h"
#include "mgl_buffer_slots.h"
#include "mgl_buffer_plan.h"
#include "mgl_tess_domain.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <chrono>
#include <tuple>
#include <utility>
#include <vector>

#include <mach/mach.h>
#include <Block.h>
#include <objc/runtime.h>

extern "C"
bool mglRenderTessFactorsDiscardPatch(uint32_t gen_mode,
                                         const float* edge,
                                         const float* inside) {
    MGLTessFactorInput in;
    MGLTessNormalizedFactors n;
    if (!edge || !inside) {
        return true;
    }
    memset(&in, 0, sizeof(in));
    in.gen_mode = gen_mode;
    in.spacing = GL_EQUAL;
    memcpy(in.outer, edge, sizeof(in.outer));
    memcpy(in.inner, inside, sizeof(in.inner));
    mglTessNormalizeFactors(&in, &n);
    return n.discard != 0;
}

extern "C"
int mglRenderRepackTessFactorTriangles(
    const void* src, uint64_t src_bytes,
    void* dst, uint64_t dst_bytes,
    uint32_t patch_count) {
    const uint64_t canonical_stride = MGL_AIR_TESS_FACTOR_RECORD_BYTES;
    const uint64_t triangle_stride = MGL_AIR_TESS_FACTOR_TRI_HALF_BYTES;
    if (!src || !dst || patch_count == 0u ||
        src_bytes < (uint64_t)patch_count * canonical_stride ||
        dst_bytes < (uint64_t)patch_count * triangle_stride) {
        return -1;
    }
    const uint8_t* in_base = (const uint8_t*)src;
    uint16_t* out_all = (uint16_t*)dst;
    for (uint32_t patch = 0u; patch < patch_count; patch++) {
        const uint16_t* in =
            (const uint16_t*)(in_base + (uint64_t)patch * canonical_stride);
        uint16_t* out = out_all + patch * 4u;
        out[0] = in[0];
        out[1] = in[1];
        out[2] = in[2];
        out[3] = in[4];
    }
    return 0;
}

extern "C"
uint64_t mglRenderTessPrimitiveCount(
    const void* factors, uint64_t bytes,
    uint32_t patch_count, uint32_t tess_gen_mode,
    uint32_t instance_count) {
    if (!factors || patch_count == 0u ||
        bytes < (uint64_t)patch_count * MGL_AIR_TESS_FACTOR_RECORD_BYTES) {
        return 0u;
    }
    /* Shared domain engine: vertex stream / verts-per-primitive.
     * Callers that know TES spacing/point_mode should use
     * mglTessGeneratedPrimitiveCount instead of this EQUAL-spacing floor. */
    const uint8_t* base = (const uint8_t*)factors;
    uint64_t items = 0u;
    const uint32_t point_mode = 0u;
    for (uint32_t patch = 0u; patch < patch_count; patch++) {
        items += mglRenderTessEvalItemsPerPatch(
            base + (uint64_t)patch * MGL_AIR_TESS_FACTOR_RECORD_BYTES,
            tess_gen_mode, GL_EQUAL, point_mode);
    }
    const uint32_t vpp = tess_gen_mode == GL_ISOLINES ? 2u : 3u;
    const uint64_t prims = vpp ? items / vpp : 0u;
    if (instance_count && prims > UINT64_MAX / instance_count) {
        return UINT64_MAX;
    }
    return prims * (uint64_t)instance_count;
}

int mglRenderIndexStreamFits(uint64_t offset, uint64_t count, uint32_t elem_bytes,
                             uint64_t metal_len) {
    if (elem_bytes == 0u) {
        return 0;
    }
    if (count > UINT64_MAX / elem_bytes) {
        return 0;
    }
    const uint64_t stream = count * elem_bytes;
    if (offset > UINT64_MAX - stream) {
        return 0;
    }
    return offset + stream <= metal_len ? 1 : 0;
}

int mglRenderDrawModeNeedsEmulate(uint32_t mode) {
    return mode == GL_TRIANGLE_FAN || mode == GL_LINE_LOOP || mode == GL_QUADS
               ? 1
               : 0;
}

int mglRenderIndexTypeIsU8(uint32_t type) {
    return type == GL_UNSIGNED_BYTE ? 1 : 0;
}

int mglRenderDrawModeIsTriangles(uint32_t mode) {
    return mode == GL_TRIANGLES ? 1 : 0;
}

int mglRenderDrawModeIsTriangleStrip(uint32_t mode) {
    return mode == GL_TRIANGLE_STRIP ? 1 : 0;
}

int mglRenderIndexTypeIsU16(uint32_t type) {
    return type == GL_UNSIGNED_SHORT ? 1 : 0;
}

int mglRenderIndexTypeIsU32(uint32_t type) {
    return type == GL_UNSIGNED_INT ? 1 : 0;
}

int mglRenderShaderResourceIndexValid(int spvc_type, uint32_t index,
                                      uint32_t count) {
    return spvc_type >= 0 && spvc_type < MGL_MAX_SHADER_RESOURCES &&
                   index < count
               ? 1
               : 0;
}

extern "C"
uint32_t mglRenderMTLPrimitiveTypeForGLMode(uint32_t mode) {
    switch (mode) {
        case GL_POINTS: return 0u;          /* MTLPrimitiveTypePoint */
        case GL_LINES: return 1u;           /* MTLPrimitiveTypeLine */
        case GL_LINE_STRIP: return 2u;      /* MTLPrimitiveTypeLineStrip */
        case GL_TRIANGLES: return 3u;       /* MTLPrimitiveTypeTriangle */
        case GL_TRIANGLE_STRIP: return 4u;  /* MTLPrimitiveTypeTriangleStrip */
        /* LINE_LOOP / adjacency / fan / quads / patches route elsewhere */
        default: return 0xFFFFFFFFu;        /* err */
    }
}

extern "C"
uint32_t mglRenderMTLIndexTypeForGLType(uint32_t gl_type) {
    switch (gl_type) {
        case GL_UNSIGNED_BYTE:
        case GL_UNSIGNED_SHORT:
            return 0u;                      /* MTLIndexTypeUInt16 */
        case GL_UNSIGNED_INT:
            return 1u;                      /* MTLIndexTypeUInt32 */
        default:
            return 0xFFFFFFFFu;             /* err */
    }
}

extern "C"
uint32_t mglRenderPrimitiveVertexCountForMode(uint32_t mode) {
    switch (mode) {
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
            return 3u;
        case GL_LINES:
        case GL_LINE_STRIP:
        case GL_LINE_LOOP:
            return 2u;
        case GL_QUADS:
            return 4u;
        case GL_POINTS:
        default:
            return 1u;
    }
}

extern "C"
uint32_t mglRenderTessRoundLevelForSpacing(uint32_t spacing,
                                              uint32_t ceil_level) {
    return mglTessRoundLevelForSpacing(spacing, ceil_level);
}

extern "C"
uint32_t mglRenderTessControlPointFormat(uint64_t gl_type) {
    switch (gl_type) {
        case GL_FLOAT: return (uint32_t)MTL::VertexFormatFloat;
        case GL_FLOAT_VEC2: return (uint32_t)MTL::VertexFormatFloat2;
        case GL_FLOAT_VEC3: return (uint32_t)MTL::VertexFormatFloat3;
        case GL_FLOAT_VEC4: return (uint32_t)MTL::VertexFormatFloat4;
        case GL_INT: return (uint32_t)MTL::VertexFormatInt;
        case GL_INT_VEC2: return (uint32_t)MTL::VertexFormatInt2;
        case GL_INT_VEC3: return (uint32_t)MTL::VertexFormatInt3;
        case GL_INT_VEC4: return (uint32_t)MTL::VertexFormatInt4;
        case GL_UNSIGNED_INT:
        case GL_BOOL: return (uint32_t)MTL::VertexFormatUInt;
        case GL_UNSIGNED_INT_VEC2:
        case GL_BOOL_VEC2: return (uint32_t)MTL::VertexFormatUInt2;
        case GL_UNSIGNED_INT_VEC3:
        case GL_BOOL_VEC3: return (uint32_t)MTL::VertexFormatUInt3;
        case GL_UNSIGNED_INT_VEC4:
        case GL_BOOL_VEC4: return (uint32_t)MTL::VertexFormatUInt4;
        default: return (uint32_t)MTL::VertexFormatInvalid;
    }
}

extern "C"
uint32_t mglRenderTessControlPointLocationFormat(uint64_t gl_type) {
    /* One 16-byte location of a control-point input.  Matrix types occupy one
     * location per column, so a single location holds `rows` floats; every
     * other type is described by the component-count mapping above. */
    switch (gl_type) {
        case GL_FLOAT_MAT2:
        case GL_FLOAT_MAT3x2:
        case GL_FLOAT_MAT4x2:
            return (uint32_t)MTL::VertexFormatFloat2;
        case GL_FLOAT_MAT3:
        case GL_FLOAT_MAT2x3:
        case GL_FLOAT_MAT4x3:
            return (uint32_t)MTL::VertexFormatFloat3;
        case GL_FLOAT_MAT4:
        case GL_FLOAT_MAT2x4:
        case GL_FLOAT_MAT3x4:
            return (uint32_t)MTL::VertexFormatFloat4;
        default:
            return mglRenderTessControlPointFormat(gl_type);
    }
}



extern "C"
int mglRenderCheckedTessCaptureSize(
    int64_t count, int64_t instance_count, uint64_t stride,
    uint64_t min_stride, uint64_t* size_out, uint64_t* offset_out) {
    if (count <= 0 || instance_count <= 0 || stride < min_stride ||
        !size_out || !offset_out) {
        return -1;
    }
    const uint64_t c = (uint64_t)count;
    const uint64_t ic = (uint64_t)instance_count;
    uint64_t records;
    if (__builtin_mul_overflow(c, ic, &records) ||
        records > UINT64_MAX / stride) {
        return -1;
    }
    *size_out = records * stride;
    *offset_out = 0u;
    return 0;
}

int mglRenderDrawPrimitives(void* render_encoder,
                               uint32_t primitive_type,
                               uint64_t vertex_start,
                               uint64_t vertex_count,
                               uint64_t instance_count,
                               uint64_t base_instance) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder || instance_count == 0) return -1;
    encoder->drawPrimitives(
        static_cast<MTL::PrimitiveType>(primitive_type),
        static_cast<NS::UInteger>(vertex_start),
        static_cast<NS::UInteger>(vertex_count),
        static_cast<NS::UInteger>(instance_count),
        static_cast<NS::UInteger>(base_instance));
    return 0;
}

int mglRenderDrawIndexedPrimitives(void* render_encoder,
                                      uint32_t primitive_type,
                                      uint64_t index_count,
                                      uint32_t index_type,
                                      void* index_buffer,
                                      uint64_t index_buffer_offset,
                                      uint64_t instance_count,
                                      int64_t base_vertex,
                                      uint64_t base_instance) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::Buffer* indices = static_cast<MTL::Buffer*>(index_buffer);
    if (!encoder || !indices || instance_count == 0) return -1;
    encoder->drawIndexedPrimitives(
        static_cast<MTL::PrimitiveType>(primitive_type),
        static_cast<NS::UInteger>(index_count),
        static_cast<MTL::IndexType>(index_type), indices,
        static_cast<NS::UInteger>(index_buffer_offset),
        static_cast<NS::UInteger>(instance_count),
        static_cast<NS::Integer>(base_vertex),
        static_cast<NS::UInteger>(base_instance));
    return 0;
}

int mglRenderDrawPrimitivesIndirect(void* render_encoder,
                                       uint32_t primitive_type,
                                       void* indirect_buffer,
                                       uint64_t indirect_buffer_offset) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::Buffer* indirect = static_cast<MTL::Buffer*>(indirect_buffer);
    if (!encoder || !indirect) return -1;
    encoder->drawPrimitives(
        static_cast<MTL::PrimitiveType>(primitive_type), indirect,
        static_cast<NS::UInteger>(indirect_buffer_offset));
    return 0;
}

int mglRenderDrawIndexedPrimitivesIndirect(
    void* render_encoder,
    uint32_t primitive_type,
    uint32_t index_type,
    void* index_buffer,
    uint64_t index_buffer_offset,
    void* indirect_buffer,
    uint64_t indirect_buffer_offset) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::Buffer* indices = static_cast<MTL::Buffer*>(index_buffer);
    MTL::Buffer* indirect = static_cast<MTL::Buffer*>(indirect_buffer);
    if (!encoder || !indices || !indirect) return -1;
    encoder->drawIndexedPrimitives(
        static_cast<MTL::PrimitiveType>(primitive_type),
        static_cast<MTL::IndexType>(index_type), indices,
        static_cast<NS::UInteger>(index_buffer_offset), indirect,
        static_cast<NS::UInteger>(indirect_buffer_offset));
    return 0;
}

extern "C" void mglRenderBuildCullDistanceLayoutFromPorts(
    MGLRenderCullDistanceLayout* layout,
    const MGLRenderCullDistanceAttribPort* ports, uint32_t port_count,
    void* dummy_mtl_buffer) {
    if (!layout) {
        return;
    }
    std::memset(layout, 0, sizeof(*layout));
    if (ports) {
        for (uint32_t i = 0u; i < port_count; i++) {
            if (!ports[i].valid || !ports[i].mtl_buffer) {
                continue;
            }
            mglRenderAccumulateCullDistanceAttrib(
                layout, ports[i].mtl_buffer, ports[i].binding_offset,
                ports[i].stride, ports[i].relativeoffset);
        }
    }
    if (!layout->mtl_buffer || layout->culldist_size == 0u) {
        layout->mtl_buffer = dummy_mtl_buffer;
        layout->binding_offset = 0;
        layout->stride = 4u;
        layout->first_relative_offset = 0;
        layout->culldist_size = 0u;
    }
}

extern "C" uint32_t mglRenderCullDistanceLayoutOffset(
    const MGLRenderCullDistanceLayout* layout) {
    if (!layout) {
        return 0u;
    }
    const int64_t rel = layout->first_relative_offset >= 0
                            ? layout->first_relative_offset
                            : 0;
    return (uint32_t)(layout->binding_offset + rel);
}

extern "C" void mglRenderFillCullDistanceEmuParams(
    uint32_t prim_vertex_count, uint32_t first_vertex,
    const uint32_t* explicit_vertices, uint32_t explicit_vertex_count,
    uint32_t culldist_offset, uint32_t vertex_stride, uint32_t culldist_size,
    uint32_t first_instance, uint32_t instance_stride,
    MGLCullDistanceEmuParams* out) {
    if (!out) {
        return;
    }
    std::memset(out, 0, sizeof(*out));
    if (explicit_vertex_count > 4u) {
        explicit_vertex_count = 4u;
    }
    out->prim_vertex_count = prim_vertex_count;
    out->culldist_offset = culldist_offset;
    out->vertex_stride = vertex_stride;
    out->culldist_size = culldist_size;
    out->first_vertex = first_vertex;
    out->explicit_vertex_count = explicit_vertex_count;
    if (explicit_vertices && explicit_vertex_count > 0u) {
        std::memcpy(out->explicit_vertices, explicit_vertices,
                    explicit_vertex_count * sizeof(uint32_t));
    }
    out->first_instance = first_instance;
    out->instance_stride = instance_stride;
}

extern "C" int mglRenderCullDistanceCaptureBytes(uint32_t first, uint32_t count,
                                                uint32_t instance_count,
                                                uint64_t* out_bytes) {
    if (!out_bytes || count == 0u || instance_count == 0u) {
        return -1;
    }
    const uint64_t endVertex = (uint64_t)first + (uint64_t)count;
    const uint64_t lastCaptureIndex =
        (uint64_t)(instance_count - 1u) * (uint64_t)count + endVertex;
    if (endVertex == 0u || lastCaptureIndex == 0u ||
        lastCaptureIndex > SIZE_MAX / 32u) {
        return -1;
    }
    *out_bytes = lastCaptureIndex * 32u;
    return 0;
}

int mglRenderDrawPatches(void* render_encoder,
                            uint64_t control_point_count,
                            uint64_t patch_start,
                            uint64_t patch_count,
                            void* patch_index_buffer,
                            uint64_t patch_index_buffer_offset,
                            uint64_t instance_count,
                            uint64_t base_instance) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder || control_point_count == 0 || instance_count == 0) return -1;
    encoder->drawPatches(
        static_cast<NS::UInteger>(control_point_count),
        static_cast<NS::UInteger>(patch_start),
        static_cast<NS::UInteger>(patch_count),
        static_cast<MTL::Buffer*>(patch_index_buffer),
        static_cast<NS::UInteger>(patch_index_buffer_offset),
        static_cast<NS::UInteger>(instance_count),
        static_cast<NS::UInteger>(base_instance));
    return 0;
}

int mglRenderDrawIndexedPatches(void* render_encoder,
                                   uint64_t control_point_count,
                                   uint64_t patch_start,
                                   uint64_t patch_count,
                                   void* patch_index_buffer,
                                   uint64_t patch_index_buffer_offset,
                                   void* control_point_index_buffer,
                                   uint64_t control_point_index_buffer_offset,
                                   uint64_t instance_count,
                                   uint64_t base_instance) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder || patch_count == 0 || instance_count == 0 ||
        !control_point_index_buffer) return -1;
    encoder->drawIndexedPatches(
        static_cast<NS::UInteger>(control_point_count),
        static_cast<NS::UInteger>(patch_start),
        static_cast<NS::UInteger>(patch_count),
        static_cast<MTL::Buffer*>(patch_index_buffer),
        static_cast<NS::UInteger>(patch_index_buffer_offset),
        static_cast<MTL::Buffer*>(control_point_index_buffer),
        static_cast<NS::UInteger>(control_point_index_buffer_offset),
        static_cast<NS::UInteger>(instance_count),
        static_cast<NS::UInteger>(base_instance));
    return 0;
}

int mglRenderSetIndirectDrawIndexed(void* indirect_command,
                                       uint32_t primitive_type,
                                       uint64_t index_count,
                                       uint32_t index_type,
                                       void* index_buffer,
                                       uint64_t index_buffer_offset,
                                       uint64_t instance_count,
                                       int64_t base_vertex,
                                       uint64_t base_instance) {
    MTL::IndirectRenderCommand* command =
        static_cast<MTL::IndirectRenderCommand*>(indirect_command);
    MTL::Buffer* indices = static_cast<MTL::Buffer*>(index_buffer);
    if (!command || !indices || instance_count == 0) return -1;
    command->drawIndexedPrimitives(
        static_cast<MTL::PrimitiveType>(primitive_type), index_count,
        static_cast<MTL::IndexType>(index_type), indices,
        index_buffer_offset, instance_count,
        static_cast<NS::Integer>(base_vertex), base_instance);
    return 0;
}

int mglRenderSetIndirectDraw(void* indirect_command,
                                uint32_t primitive_type,
                                uint64_t vertex_start,
                                uint64_t vertex_count,
                                uint64_t instance_count,
                                uint64_t base_instance) {
    MTL::IndirectRenderCommand* command =
        static_cast<MTL::IndirectRenderCommand*>(indirect_command);
    if (!command || instance_count == 0) return -1;
    command->drawPrimitives(static_cast<MTL::PrimitiveType>(primitive_type),
                            vertex_start, vertex_count, instance_count,
                            base_instance);
    return 0;
}

int mglRenderGetCullDistanceIndexPrimitive(MGLCullDistanceIndexPlan * owner, uint64_t primitive_index, MGLRenderCullDistancePrimitive* primitive_out) {
    mgl::CullDistanceIndexPlan* plan =
        reinterpret_cast<mgl::CullDistanceIndexPlan*>(static_cast<void*>(owner));
    if (!plan || !primitive_out ||
        primitive_index >= plan->primitives.size()) {
        return -1;
    }
    *primitive_out = plan->primitives[primitive_index];
    return 0;
}

void mglRenderDestroyCullDistanceIndexPlan(MGLCullDistanceIndexPlan ** owner) {
    if (!owner || !*owner) return;
    delete reinterpret_cast<mgl::CullDistanceIndexPlan*>(*owner);
    *owner = nullptr;
}

extern "C"
uint32_t mglRenderGLIndexElementSize(uint64_t gl_index_type) {
    if (gl_index_type == GL_UNSIGNED_BYTE) return 1u;
    if (gl_index_type == GL_UNSIGNED_SHORT) return 2u;
    if (gl_index_type == GL_UNSIGNED_INT) return 4u;
    return 0u;
}

extern "C"
uint32_t mglRenderReadGLIndexValue(const uint8_t* bytes, uint32_t elem_width,
                                      uint64_t element_index) {
    if (!bytes || elem_width == 0u) {
        return 0u;
    }
    if (elem_width == 1u) {
        uint8_t v = 0u;
        memcpy(&v, bytes + element_index, sizeof(v));
        return (uint32_t)v;
    }
    if (elem_width == 2u) {
        uint16_t v = 0u;
        memcpy(&v, bytes + (element_index * 2u), sizeof(v));
        return (uint32_t)v;
    }
    if (elem_width == 4u) {
        uint32_t v = 0u;
        memcpy(&v, bytes + (element_index * 4u), sizeof(v));
        return v;
    }
    return 0u;
}

extern "C"
int mglRenderDrawModeProducesPolygons(uint64_t gl_mode) {
    switch (gl_mode) {
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
        case GL_QUADS:
            return 1;
        default:
            return 0;
    }
}

extern "C"
int mglRenderPrimitiveModeHasDrawableSegment(uint64_t gl_mode,
                                                uint64_t index_count) {
    switch (gl_mode) {
        case GL_POINTS:
            return index_count >= 1u ? 1 : 0;
        case GL_LINES:
        case GL_LINE_STRIP:
        case GL_LINE_LOOP:
            return index_count >= 2u ? 1 : 0;
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
            return index_count >= 3u ? 1 : 0;
        case GL_QUADS:
            return index_count >= 4u ? 1 : 0;
        default:
            return index_count > 0u ? 1 : 0;
    }
}

extern "C"
uint64_t mglRenderQuadTriangleIndexCount(uint64_t source_vertex_count) {
    const uint64_t quad_count = source_vertex_count / 4u;
    if (quad_count > (uint64_t)(SIZE_MAX / 6u)) {
        return 0u;
    }
    return quad_count * 6u;
}

extern "C"
int mglRenderPrimitiveRestartFixedIndex(uint64_t gl_index_type, uint32_t* out) {
    if (!out) {
        return -1;
    }
    switch (gl_index_type) {
        case GL_UNSIGNED_BYTE: *out = 0xffu; return 1;
        case GL_UNSIGNED_SHORT: *out = 0xffffu; return 1;
        case GL_UNSIGNED_INT: *out = 0xffffffffu; return 1;
        default: return 0;
    }
}

extern "C"
int mglRenderPlanCullDistanceElementRange(
    const uint8_t* bytes, uint32_t elem_width, uint32_t count,
    int restart_enabled, uint32_t restart_index, int32_t base_vertex,
    int32_t* out_first, uint32_t* out_count) {
    if (!out_first || !out_count) {
        return -1;
    }
    *out_first = 0;
    *out_count = 0u;
    uint32_t scan_min = 0u, scan_max = 0u;
    int scan_valid = 0;
    if (mglRenderScanIndexRangeIgnoringRestart(
            bytes, elem_width, count, restart_enabled, restart_index,
            &scan_min, &scan_max, &scan_valid) != 0 || !scan_valid) {
        return -1;
    }
    const int64_t first = (int64_t)scan_min + (int64_t)base_vertex;
    const int64_t last = (int64_t)scan_max + (int64_t)base_vertex;
    if (first < 0 || last < first || last > INT32_MAX) {
        return -1;
    }
    const uint64_t vertex_count = (uint64_t)(last - first) + 1u;
    if (vertex_count > INT32_MAX) {
        return -1;
    }
    *out_first = (int32_t)first;
    *out_count = (uint32_t)vertex_count;
    return 0;
}

extern "C"
int mglRenderComputePreparedIndexByteOffset(
    uint64_t gl_index_type, uint64_t gl_byte_offset,
    uint64_t* out_prepared_offset) {
    if (!out_prepared_offset) {
        return -1;
    }
    if (gl_index_type == GL_UNSIGNED_BYTE) {
        // GL_UNSIGNED_BYTE indices expand to UInt16 (2 bytes each).
        if (gl_byte_offset > (uint64_t)(SIZE_MAX / sizeof(uint16_t))) {
            return -1;
        }
        *out_prepared_offset = gl_byte_offset * sizeof(uint16_t);
        return 0;
    }
    *out_prepared_offset = gl_byte_offset;
    return 0;
}

extern "C"
int mglRenderComputeIndexByteOffset(
    uint64_t base_byte_offset, uint64_t first_element, uint64_t index_stride,
    uint64_t* out_byte_offset) {
    if (!out_byte_offset || index_stride == 0u) {
        return -1;
    }
    if (first_element > (uint64_t)SIZE_MAX / index_stride) {
        return -1;
    }
    const uint64_t relative = first_element * index_stride;
    if (base_byte_offset > (uint64_t)SIZE_MAX - relative) {
        return -1;
    }
    *out_byte_offset = base_byte_offset + relative;
    return 0;
}

int mglRenderCreateCullDistanceIndexPlan(void* device, const void* source_indices, uint32_t source_index_type, uint64_t source_index_count, uint32_t draw_mode, int primitive_restart_enabled, uint32_t primitive_restart_index, int64_t base_vertex, int polygon_line_mode, MGLCullDistanceIndexPlan ** owner_out, void** index_buffer_out, uint64_t* primitive_count_out) {
    if (owner_out) *owner_out = nullptr;
    if (index_buffer_out) *index_buffer_out = nullptr;
    if (primitive_count_out) *primitive_count_out = 0;
    if (!source_indices || !owner_out || !index_buffer_out ||
        !primitive_count_out || source_index_count == 0 ||
        source_index_count > static_cast<uint64_t>(SIZE_MAX)) {
        return -1;
    }

    const uint8_t* bytes = static_cast<const uint8_t*>(source_indices);
    std::vector<uint32_t> source;
    std::vector<uint32_t> expanded;
    std::unique_ptr<mgl::CullDistanceIndexPlan> plan(
        new (std::nothrow) mgl::CullDistanceIndexPlan());
    if (!plan) return -1;
    try {
        source.reserve(static_cast<size_t>(source_index_count));
        expanded.reserve(static_cast<size_t>(source_index_count));
        plan->primitives.reserve(static_cast<size_t>(source_index_count));
        for (uint64_t index = 0; index < source_index_count; ++index) {
            uint32_t value = 0;
            if (!mgl::readCullDistanceSourceIndex(
                    bytes, source_index_type, index, value)) {
                return -1;
            }
            source.push_back(value);
        }

        size_t segmentBegin = 0;
        for (size_t index = 0; index <= source.size(); ++index) {
            const bool atEnd = index == source.size();
            const bool atRestart = !atEnd && primitive_restart_enabled &&
                                   source[index] == primitive_restart_index;
            if (!atEnd && !atRestart) continue;
            if (index > segmentBegin &&
                !mgl::appendCullDistanceSegment(
                    source, segmentBegin, index, draw_mode,
                    polygon_line_mode != 0, base_vertex, expanded,
                    plan->primitives)) {
                return -1;
            }
            segmentBegin = index + 1u;
        }
    } catch (...) {
        return -1;
    }

    if (!expanded.empty()) {
        if (expanded.size() > SIZE_MAX / sizeof(uint32_t)) return -1;
        MTL::Device* metalDevice = static_cast<MTL::Device*>(device);
        if (!metalDevice) {
            mgl::Renderer& renderer = mgl::renderer();
            std::lock_guard<std::mutex> lock(renderer.mutex);
            metalDevice = renderer.device;
            if (!metalDevice) return -1;
            plan->indexBuffer = metalDevice->newBuffer(
                expanded.data(), expanded.size() * sizeof(uint32_t),
                MTL::ResourceStorageModeShared);
        } else {
            plan->indexBuffer = metalDevice->newBuffer(
                expanded.data(), expanded.size() * sizeof(uint32_t),
                MTL::ResourceStorageModeShared);
        }
        if (!plan->indexBuffer) return -1;
        plan->indexBuffer->setLabel(NS::String::string(
            "MGL CullDistance expanded indices", NS::UTF8StringEncoding));
    }

    *index_buffer_out = plan->indexBuffer;
    *primitive_count_out = plan->primitives.size();
    *owner_out = reinterpret_cast<MGLCullDistanceIndexPlan*>(plan.release());
    return 0;
}

extern "C"
int mglRenderScanIndexRangeIgnoringRestart(
    const uint8_t* bytes, uint32_t elem_width, uint32_t count,
    int restart_enabled, uint32_t restart_index,
    uint32_t* out_min, uint32_t* out_max, int* out_valid) {
    if (!bytes || count == 0u || !out_min || !out_max || !out_valid) {
        return -1;
    }
    const int w = (elem_width == 1u) ? 1 : (elem_width == 2u ? 2 : 4);
    uint32_t min_index = UINT32_MAX;
    uint32_t max_index = 0u;
    const int has_restart = restart_enabled ? 1 : 0;
    for (uint32_t i = 0u; i < count; i++) {
        const uint32_t v = MGLRenderReadIndexBytes(bytes, w, i);
        if (has_restart && v == restart_index) {
            continue;
        }
        if (v < min_index) min_index = v;
        if (v > max_index) max_index = v;
    }
    *out_min = min_index;
    *out_max = max_index;
    *out_valid = (min_index <= max_index) ? 1 : 0;
    return 0;
}

static MGLTessFactorInput tessDomainInput(const void *factor_record,
    uint32_t gen_mode, uint32_t spacing, uint32_t point_mode, uint32_t winding)
{
    MGLTessFactorInput in = {};
    in.gen_mode = gen_mode;
    in.spacing = spacing;
    in.winding = winding;
    in.point_mode = point_mode ? 1u : 0u;
    const uint8_t *exact = (const uint8_t *)factor_record +
        MGL_AIR_TESS_FACTOR_EXACT_FLOAT_OFFSET;
    memcpy(in.outer, exact, sizeof(in.outer));
    memcpy(in.inner, exact + sizeof(in.outer), sizeof(in.inner));
    return in;
}

static int mglCullDistanceArraySplitCount(uint32_t draw_mode, uint64_t count,
                                          uint32_t* out_n) {
    if (!out_n) {
        return -1;
    }
    *out_n = 0u;
    if (count > UINT32_MAX) {
        return -1;
    }
    switch (draw_mode) {
    case GL_TRIANGLE_STRIP:
    case GL_TRIANGLE_FAN:
        if (count < 3u) {
            return 1;
        }
        *out_n = (uint32_t)(count - 2u);
        return 0;
    case GL_LINE_STRIP:
        if (count < 2u) {
            return 1;
        }
        *out_n = (uint32_t)(count - 1u);
        return 0;
    case GL_LINE_LOOP:
        if (count < 2u) {
            return 1;
        }
        *out_n = (uint32_t)count;
        return 0;
    default:
        return 1;
    }
}

int mglRenderFillCullDistanceArrayPrimitives(
    uint32_t draw_mode, int32_t first, uint64_t count,
    MGLRenderCullDistancePrimitive* out, uint32_t cap, uint32_t* out_count) {
    if (!out_count) {
        return -1;
    }
    uint32_t n = 0u;
    const int kind = mglCullDistanceArraySplitCount(draw_mode, count, &n);
    *out_count = 0u;
    if (kind != 0) {
        return kind;
    }
    if (!out) {
        *out_count = n;
        return 0;
    }
    if (n > cap) {
        return -1;
    }
    const uint32_t base = (uint32_t)first;
    for (uint32_t p = 0u; p < n; p++) {
        MGLRenderCullDistancePrimitive prim = {};
        if (draw_mode == GL_TRIANGLE_STRIP) {
            prim.vertices[0] = base + p;
            prim.vertices[1] = base + p + 1u;
            prim.vertices[2] = base + p + 2u;
            prim.vertex_count = 3u;
            prim.primitive_type = 3u; /* MGL_DRAW_PRIMITIVE_TRIANGLE */
            prim.index_count = 3u;
            prim.index_buffer_offset = (uint64_t)p * 3u * sizeof(uint32_t);
        } else if (draw_mode == GL_TRIANGLE_FAN) {
            prim.vertices[0] = base;
            prim.vertices[1] = base + p + 1u;
            prim.vertices[2] = base + p + 2u;
            prim.vertex_count = 3u;
            prim.primitive_type = 3u;
            prim.index_count = 3u;
            prim.index_buffer_offset = (uint64_t)p * 3u * sizeof(uint32_t);
        } else if (draw_mode == GL_LINE_STRIP) {
            prim.vertices[0] = base + p;
            prim.vertex_count = 0u;
            prim.primitive_type = 1u; /* MGL_DRAW_PRIMITIVE_LINE */
            prim.index_count = 0u;
            prim.index_buffer_offset = 0u;
        } else {
            prim.vertices[0] = base + p;
            prim.vertices[1] = base + ((p + 1u) % (uint32_t)count);
            prim.vertex_count = 2u;
            prim.primitive_type = 1u;
            prim.index_count = 2u;
            prim.index_buffer_offset = (uint64_t)p * sizeof(uint32_t);
        }
        out[p] = prim;
    }
    *out_count = n;
    return 0;
}

static MTL::Buffer* mglCullDistanceNewIndexBuffer(void* device,
                                                 const uint32_t* data,
                                                 size_t count) {
    if (!data || count == 0u) {
        return nullptr;
    }
    MTL::Device* metalDevice = static_cast<MTL::Device*>(device);
    if (!metalDevice) {
        mgl::Renderer& renderer = mgl::renderer();
        std::lock_guard<std::mutex> lock(renderer.mutex);
        metalDevice = renderer.device;
        if (!metalDevice) {
            return nullptr;
        }
        MTL::Buffer* buffer = metalDevice->newBuffer(
            data, count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        if (buffer) {
            buffer->setLabel(NS::String::string(
                "MGL CullDistance expanded indices", NS::UTF8StringEncoding));
        }
        return buffer;
    }
    MTL::Buffer* buffer = metalDevice->newBuffer(
        data, count * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    if (buffer) {
        buffer->setLabel(NS::String::string(
            "MGL CullDistance expanded indices", NS::UTF8StringEncoding));
    }
    return buffer;
}

int mglRenderCreateCullDistanceArrayPlan(void* device, uint32_t draw_mode, int32_t first, uint64_t count, MGLCullDistanceIndexPlan ** owner_out, void** index_buffer_out, uint64_t* primitive_count_out) {
    if (owner_out) *owner_out = nullptr;
    if (index_buffer_out) *index_buffer_out = nullptr;
    if (primitive_count_out) *primitive_count_out = 0;
    if (!owner_out || !index_buffer_out || !primitive_count_out) {
        return -1;
    }
    uint32_t n = 0u;
    const int kind =
        mglRenderFillCullDistanceArrayPrimitives(draw_mode, first, count,
                                                 nullptr, 0u, &n);
    if (kind != 0) {
        return kind;
    }
    std::unique_ptr<mgl::CullDistanceIndexPlan> plan(
        new (std::nothrow) mgl::CullDistanceIndexPlan());
    if (!plan) {
        return -1;
    }
    try {
        plan->primitives.resize(n);
    } catch (...) {
        return -1;
    }
    uint32_t filled = 0u;
    if (mglRenderFillCullDistanceArrayPrimitives(
            draw_mode, first, count, plan->primitives.data(), n, &filled) !=
            0 ||
        filled != n) {
        return -1;
    }
    std::vector<uint32_t> expanded;
    if (draw_mode == GL_TRIANGLE_STRIP || draw_mode == GL_TRIANGLE_FAN) {
        uint32_t* raw = nullptr;
        uint64_t raw_count = 0u;
        const int expand =
            draw_mode == GL_TRIANGLE_STRIP
                ? mglRenderExpandTriangleStripArrayIndices((uint32_t)count,
                                                           &raw, &raw_count)
                : mglRenderExpandTriangleFanArrayIndices((uint32_t)count, &raw,
                                                         &raw_count);
        if (expand != 0 || !raw || raw_count == 0u) {
            std::free(raw);
            return -1;
        }
        const uint32_t base = (uint32_t)first;
        try {
            expanded.resize(static_cast<size_t>(raw_count));
            for (uint64_t i = 0u; i < raw_count; i++) {
                expanded[static_cast<size_t>(i)] = raw[i] + base;
            }
        } catch (...) {
            std::free(raw);
            return -1;
        }
        std::free(raw);
    } else if (draw_mode == GL_LINE_LOOP) {
        uint32_t* raw = nullptr;
        uint64_t raw_count = 0u;
        if (mglRenderExpandLineLoopArrayIndices((uint32_t)first, (uint32_t)count,
                                                &raw, &raw_count) != 0 ||
            !raw) {
            std::free(raw);
            return -1;
        }
        try {
            expanded.assign(raw, raw + static_cast<size_t>(raw_count));
        } catch (...) {
            std::free(raw);
            return -1;
        }
        std::free(raw);
    }
    if (!expanded.empty()) {
        if (expanded.size() > SIZE_MAX / sizeof(uint32_t)) {
            return -1;
        }
        plan->indexBuffer = mglCullDistanceNewIndexBuffer(
            device, expanded.data(), expanded.size());
        if (!plan->indexBuffer) {
            return -1;
        }
    }
    *index_buffer_out = plan->indexBuffer;
    *primitive_count_out = plan->primitives.size();
    *owner_out = reinterpret_cast<MGLCullDistanceIndexPlan*>(plan.release());
    return 0;
}

static bool mglRenderProgramUsesVertexAttrib(const Program* program,
                                            uint32_t attrib) {
    if (!program || attrib >= MAX_ATTRIBS) {
        return false;
    }
    const MGLShaderResourceList* inputs =
        &program->shader_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES];
    if (!inputs->list || inputs->count == 0) {
        return false;
    }
    for (GLuint i = 0; i < inputs->count; i++) {
        const GLuint location = inputs->list[i].location;
        if (location == attrib) {
            return true;
        }
        const GLuint span =
            mglAIRVaryingLocationSpan(inputs->list[i].gl_type,
                                      inputs->list[i].gl_array_size);
        if (span > 1u && attrib >= location && attrib < location + span) {
            return true;
        }
        if (location == 0xffffffffu && i == attrib) {
            return true;
        }
    }
    return false;
}

extern "C" uint32_t mglRenderCollectCullDistanceAttribs(const Program* program,
                                                       uint32_t* out,
                                                       uint32_t cap) {
    uint32_t n = 0u;
    if (!program) {
        return 0u;
    }
    for (uint32_t attrib = 0u; attrib < MAX_ATTRIBS; attrib++) {
        if (!mglRenderProgramUsesVertexAttrib(program, attrib)) {
            continue;
        }
        if (!mglRenderIsCullDistanceAttribName(
                mglRenderVertexAttribName(program, attrib))) {
            continue;
        }
        if (out && n < cap) {
            out[n] = attrib;
        }
        n++;
    }
    return n;
}

extern "C"
uint32_t mglRenderTessEvalItemsPerPatch(
    const void* factor_record, uint32_t gen_mode, uint32_t spacing,
    uint32_t point_mode) {
    if (!factor_record) return 0;
    const MGLTessFactorInput in = tessDomainInput(
        factor_record, gen_mode, spacing, point_mode, GL_CCW);
    return mglTessDomainVertexCount(&in);
}

extern "C"
uint32_t mglRenderSeedTessDomain(const void *factor_record,
    uint32_t gen_mode, uint32_t spacing, uint32_t point_mode, uint32_t winding,
    void *records, uint32_t count, uint32_t stride)
{
    if (!factor_record) return 0;
    const MGLTessFactorInput in = tessDomainInput(
        factor_record, gen_mode, spacing, point_mode, winding);
    return mglTessGenerateDomainStrided(&in, records, count, stride);
}

extern "C"
int mglRenderExpandQuadElementLineIndices(
    const uint8_t* bytes, uint32_t elem_width, uint32_t quad_count,
    uint32_t** out_indices, uint64_t* out_count) {
    if (!bytes || quad_count == 0u || !out_indices || !out_count) {
        return -1;
    }
    const uint64_t need = (uint64_t)quad_count * 8u;
    if (need > (uint64_t)(UINT32_MAX / sizeof(uint32_t))) {
        return -1;
    }
    uint32_t* const dst = (uint32_t*)malloc((size_t)need * sizeof(uint32_t));
    if (!dst) {
        return -1;
    }
    const int w = (elem_width == 1u) ? 1 : (elem_width == 2u ? 2 : 4);
    for (uint32_t q = 0u; q < quad_count; q++) {
        const uint32_t src = q * 4u;
        const uint32_t d = q * 8u;
        const uint32_t i0 = MGLRenderReadIndexBytes(bytes, w, src + 0u);
        const uint32_t i1 = MGLRenderReadIndexBytes(bytes, w, src + 1u);
        const uint32_t i2 = MGLRenderReadIndexBytes(bytes, w, src + 2u);
        const uint32_t i3 = MGLRenderReadIndexBytes(bytes, w, src + 3u);
        dst[d+0]=i0; dst[d+1]=i1; dst[d+2]=i1; dst[d+3]=i2;
        dst[d+4]=i2; dst[d+5]=i3; dst[d+6]=i3; dst[d+7]=i0;
    }
    *out_indices = dst;
    *out_count = need;
    return 0;
}

extern "C"
int mglRenderExpandQuadElementIndices(
    const uint8_t* bytes, uint32_t elem_width, uint32_t quad_count,
    uint32_t** out_indices, uint64_t* out_count) {
    if (!bytes || quad_count == 0u || !out_indices || !out_count) {
        return -1;
    }
    const uint64_t need = (uint64_t)quad_count * 6u;
    if (need > (uint64_t)(UINT32_MAX / sizeof(uint32_t))) {
        return -1;
    }
    uint32_t* const dst = (uint32_t*)malloc((size_t)need * sizeof(uint32_t));
    if (!dst) {
        return -1;
    }
    const int w = (elem_width == 1u) ? 1 : (elem_width == 2u ? 2 : 4);
    for (uint32_t q = 0u; q < quad_count; q++) {
        const uint32_t src = q * 4u;
        const uint32_t d = q * 6u;
        const uint32_t i0 = MGLRenderReadIndexBytes(bytes, w, src + 0u);
        const uint32_t i1 = MGLRenderReadIndexBytes(bytes, w, src + 1u);
        const uint32_t i2 = MGLRenderReadIndexBytes(bytes, w, src + 2u);
        const uint32_t i3 = MGLRenderReadIndexBytes(bytes, w, src + 3u);
        dst[d+0] = i0;
        dst[d+1] = i1;
        dst[d+2] = i2;
        dst[d+3] = i0;
        dst[d+4] = i2;
        dst[d+5] = i3;
    }
    *out_indices = dst;
    *out_count = need;
    return 0;
}

extern "C"
int mglRenderExpandTriangleStripIndices(
    const uint8_t* bytes, uint32_t elem_width, uint32_t count,
    uint32_t** out_indices, uint64_t* out_count) {
    if (!bytes || count < 3u || !out_indices || !out_count) {
        return -1;
    }
    const int w = (elem_width == 1u) ? 1 : (elem_width == 2u ? 2 : 4);
    const uint32_t n = count - 2u;
    const uint64_t need = (uint64_t)n * 3u;
    if (need > (uint64_t)(UINT32_MAX / sizeof(uint32_t))) {
        return -1;
    }
    uint32_t* const dst = (uint32_t*)malloc((size_t)need * sizeof(uint32_t));
    if (!dst) {
        return -1;
    }
    for (uint32_t t = 0u; t < n; t++) {
        const uint32_t first = t + (t & 1u);
        const uint32_t second = t + ((t & 1u) ? 0u : 1u);
        dst[t*3u+0u] = MGLRenderReadIndexBytes(bytes, w, first);
        dst[t*3u+1u] = MGLRenderReadIndexBytes(bytes, w, second);
        dst[t*3u+2u] = MGLRenderReadIndexBytes(bytes, w, t + 2u);
    }
    *out_indices = dst;
    *out_count = need;
    return 0;
}

extern "C"
int mglRenderExpandLineLoopIndices(
    const uint8_t* bytes, uint32_t elem_width, uint32_t count,
    uint32_t** out_indices, uint64_t* out_count) {
    if (!bytes || count < 2u || !out_indices || !out_count) {
        return -1;
    }
    const uint64_t need = (uint64_t)count + 1u;
    uint32_t* const dst = (uint32_t*)malloc((size_t)need * sizeof(uint32_t));
    if (!dst) {
        return -1;
    }
    const int w = (elem_width == 1u) ? 1 : (elem_width == 2u ? 2 : 4);
    for (uint32_t i = 0u; i < count; i++) {
        dst[i] = MGLRenderReadIndexBytes(bytes, w, i);
    }
    dst[count] = dst[0];
    *out_indices = dst;
    *out_count = need;
    return 0;
}
