/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
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
 * mgl_shader_abi.h
 * MGL - pure C ABI boundary between the C/ObjC side and the C++ AIR
 * backend (see docs/AIR_SHADER_BACKEND_DESIGN.md).  C/ObjC code never
 * sees LLVM types; it hands GLSL source to mglShaderCompileGLSL and
 * receives a self-contained .metallib byte blob for newLibraryWithData.
 */

#ifndef MGL_SHADER_ABI_H
#define MGL_SHADER_ABI_H

#include <stddef.h>
#include <stdint.h>

typedef struct GLMContextRec_t *GLMContext;

#include "mgl_types_program.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum MGLShaderStage {
    MGL_STAGE_VERTEX = 0,
    MGL_STAGE_FRAGMENT,
    MGL_STAGE_COMPUTE,
    /* M3: tessellation + geometry (Metal 4: native tessellation factors +
     * post-tessellation vertex; GS via compute expansion). */
    MGL_STAGE_TESS_CONTROL,
    MGL_STAGE_TESS_EVALUATION,
    MGL_STAGE_GEOMETRY,
} MGLShaderStage;

/* Stage execution metadata needed by the renderer.  Values use GL enums at
 * the C ABI boundary so Program does not depend on parser-private enums. */
typedef struct MGLAIRStageInfo {
    uint32_t tess_control_output_vertices;
    uint32_t tess_patch_vertices;   /* TES: control points per patch
                                     * (TCS output or glPatchParameteri) */
    uint32_t tess_gen_mode;
    uint32_t tess_gen_spacing;
    uint32_t tess_gen_vertex_order;
    uint32_t tess_gen_point_mode;
    uint32_t geometry_input_type;
    uint32_t geometry_output_type;
    uint32_t geometry_vertices_out;
    uint32_t geometry_max_vertices_specified; /* 1 if layout(max_vertices) present */
    uint32_t geometry_invocations;
    uint32_t uses_cull_distance;
    uint32_t cull_distance_count;
    uint32_t needs_runtime_array_size_buffer;
    /* GS multi-stream (GL 4.6 §11.1.3.4): stream 0 is rasterized; streams
     * 1..3 feed transform feedback only and require points output.  The
     * per-stream XFB capture record is position (16B) + that stream's
     * varyings in ascending location order (16B each). */
    uint32_t gs_stream_count;            /* number of streams used (1..4) */
    uint32_t gs_stream_varying_count[4];
    uint32_t gs_stream_xfb_stride[4];
} MGLAIRStageInfo;

/* Fixed inter-stage record shared by VS capture, TCS, TES and the GS compute
 * expansion path.  Keeping the offsets in the public C ABI prevents the AIR
 * backend and renderer from silently drifting apart when built separately. */
enum {
    MGL_AIR_PER_VERTEX_POSITION_OFFSET = 0,
    MGL_AIR_PER_VERTEX_POINT_SIZE_OFFSET = 16,
    MGL_AIR_PER_VERTEX_CULL_DISTANCE_OFFSET = 20,
    MGL_AIR_PER_VERTEX_CULL_DISTANCE_COUNT = 8,
    /* gl_ClipDistance (GL_MAX_CLIP_DISTANCES): plain-VS output array. */
    MGL_MAX_CLIP_DISTANCES = 8,
    /* gl_Layer / gl_ViewportIndex outputs (GS expansion): one int per
     * vertex; the rasterizing vertex stage re-emits both (GL 4.6
     * §11.1.3.5/§11.1.3.6 tie them to the same value). */
    MGL_AIR_PER_VERTEX_LAYER_OFFSET = 40,
    MGL_AIR_PER_VERTEX_VIEWPORT_INDEX_OFFSET = 44,
    /* GS multi-stream XFB: stream id stamped by EmitStreamVertex so the
     * pass-2 scatter can attribute each record to its stream (GL 4.6
     * §11.1.3.4).  Stream-0 records are identified by region, not stamp. */
    MGL_AIR_PER_VERTEX_STREAM_OFFSET = 48,
    /* GS-written gl_PrimitiveID at a dedicated offset; ferried to the
     * fragment stage through the reserved varying location below. */
    MGL_AIR_PER_VERTEX_PRIMITIVE_ID_OFFSET = 52,
    /* Reserved varying location carrying gl_PrimitiveID from the geometry
     * passthrough vertex function to the fragment shader. */
    MGL_AIR_PRIMITIVE_ID_LOCATION = 31,
    MGL_AIR_PER_VERTEX_STRIDE = 64,
};

/* Byte layout of one per-vertex record.  The kernel and the renderer
 * address every field through the MGL_AIR_PER_VERTEX_* offsets above;
 * this struct mirrors those constants for documentation and static
 * checking.
 *
 * cull_distance[5]/[6]/[7] share bytes with layer / viewport_index /
 * stream when those built-ins are used; shaders that emit six or more
 * cull distances cannot also write the overlapping built-ins. */
typedef struct MGLAIRPerVertexRecord {
    float position[4];          /* @0                                    */
    float point_size;           /* @16                                   */
    float cull_distance_lo[5];  /* @20 .. @39                            */
    int32_t layer;              /* @40  aliases cull_distance[5]         */
    int32_t viewport_index;     /* @44  aliases cull_distance[6]         */
    uint32_t stream;            /* @48  aliases cull_distance[7]         */
    float cull_distance_hi[3];  /* @52 .. @63                            */
} MGLAIRPerVertexRecord;

#include <stddef.h>
#if defined(__cplusplus)
#define MGL_AIR_VA_STATIC_ASSERT(c, m) static_assert(c, m)
#else
#define MGL_AIR_VA_STATIC_ASSERT(c, m) _Static_assert(c, m)
#endif
MGL_AIR_VA_STATIC_ASSERT(offsetof(MGLAIRPerVertexRecord, position) ==
                  MGL_AIR_PER_VERTEX_POSITION_OFFSET,
              "position offset drift");
MGL_AIR_VA_STATIC_ASSERT(offsetof(MGLAIRPerVertexRecord, point_size) ==
                  MGL_AIR_PER_VERTEX_POINT_SIZE_OFFSET,
              "point_size offset drift");
MGL_AIR_VA_STATIC_ASSERT(offsetof(MGLAIRPerVertexRecord, cull_distance_lo) ==
                  MGL_AIR_PER_VERTEX_CULL_DISTANCE_OFFSET,
              "cull_distance offset drift");
MGL_AIR_VA_STATIC_ASSERT(offsetof(MGLAIRPerVertexRecord, layer) ==
                  MGL_AIR_PER_VERTEX_LAYER_OFFSET,
              "layer offset must match the kernel's store offset");
MGL_AIR_VA_STATIC_ASSERT(offsetof(MGLAIRPerVertexRecord, viewport_index) ==
                  MGL_AIR_PER_VERTEX_VIEWPORT_INDEX_OFFSET,
              "viewport_index offset must match the kernel's store offset");
MGL_AIR_VA_STATIC_ASSERT(offsetof(MGLAIRPerVertexRecord, stream) ==
                  MGL_AIR_PER_VERTEX_STREAM_OFFSET,
              "stream offset must match the kernel's stamp offset");
MGL_AIR_VA_STATIC_ASSERT(sizeof(MGLAIRPerVertexRecord) == MGL_AIR_PER_VERTEX_STRIDE,
              "record size drift");

static inline uint32_t mglAIRPerVertexStrideForResources(
    const MGLShaderResourceList *resources)
{
    uint32_t stride = MGL_AIR_PER_VERTEX_STRIDE;
    if (!resources || !resources->list) return stride;
    for (uint32_t i = 0; i < resources->count; i++) {
        const MGLShaderResource *resource = &resources->list[i];
        if (resource->is_per_patch || resource->location >= 0x0fffffffu)
            continue;
        uint32_t end = MGL_AIR_PER_VERTEX_STRIDE +
                       (resource->location + 1u) * 16u;
        if (end > stride) stride = end;
    }
    return stride;
}

/* Parse and semantically validate one stage, then return execution metadata
 * without requiring backend code generation to be available for that stage. */
int mglAirReflectGLSLStageInfo(const char *src, int stage,
                               MGLAIRStageInfo *stage_info,
                               char *err_buf, size_t err_cap);

/* Compile a GLSL source string for one stage into a .metallib byte blob.
 *
 * On success returns 0 and sets *metallib_out to malloc'd bytes
 * (caller frees) and *size_out to its length.
 * On failure returns -1 and writes a NUL-terminated message into err_buf
 * (err_cap bytes) if err_buf is non-NULL.
 */
int mglShaderCompileGLSL(const char *src, int stage,
                         unsigned char **metallib_out, size_t *size_out,
                         char *err_buf, size_t err_cap);

/* XFB capture variant of a vertex shader: the full output record
 * (position + varyings) is written to a device buffer at Metal buffer
 * index 29 with rasterization disabled.  Returns 0 on success. */
int mglShaderCompileGLSLCapture(const char *src, unsigned char **metallib_out,
                                size_t *size_out, char *err_buf,
                                size_t err_cap);

/* Tessellation pre-pass variant of a vertex shader.  It runs as a
 * rasterization-disabled vertex function and writes MGLAIRPerVertexRecord at
 * Metal buffer index 29. */
int mglShaderCompileGLSLTessCapture(const char *src,
                                    unsigned char **metallib_out,
                                    size_t *size_out, char *err_buf,
                                    size_t err_cap);

/* Vertex pre-pass variant for exact primitive-level gl_CullDistance
 * evaluation. The capture record contains the normal vertex outputs followed
 * by float[8] cull distances and is written to buffer index 29. */
int mglShaderCompileGLSLCullDistanceCapture(
    const char *src, unsigned char **metallib_out, size_t *size_out,
    char *err_buf, size_t err_cap);

/* Compile one stage through the self-hosted frontend + AIR backend and
 * export its resource tables: metallib bytes + MGLShaderResourceList.
 * attrib_names is an optional MAX_ATTRIBS-sized array of glBindAttribLocation
 * names (index = desired location); pass NULL for no explicit bindings.
 * Returns 0 on success; lists may be NULL to skip reflection. */
int mglAirCompileGLSLWithReflect(const char *src, int stage,
                                 const char *const *attrib_names,
                                 unsigned char **metallib_out,
                                 size_t *size_out,
                                 MGLShaderResourceList lists[MGL_MAX_SHADER_RESOURCES],
                                 char *err_buf, size_t err_cap);

int mglAirCompileGLSLWithReflectInfo(
    const char *src, int stage, const char *const *attrib_names,
    unsigned char **metallib_out, size_t *size_out,
    MGLShaderResourceList lists[MGL_MAX_SHADER_RESOURCES], MGLAIRStageInfo *stage_info,
    char *err_buf, size_t err_cap);

/* mglAirCompileGLSLWithReflectInfoEx flags */
enum {
    MGL_AIR_COMPILE_HAS_GEOMETRY_SHADER = 1u << 0,
};

/* Same as above plus stage-composition flags (bit0: a geometry shader is
 * attached, which changes fragment-stage gl_PrimitiveID lowering). */
int mglAirCompileGLSLWithReflectInfoEx(
    const char *src, int stage, const char *const *attrib_names,
    unsigned char **metallib_out, size_t *size_out,
    MGLShaderResourceList lists[MGL_MAX_SHADER_RESOURCES], MGLAIRStageInfo *stage_info,
    uint32_t flags, char *err_buf, size_t err_cap);

/* Free bytes returned by mglShaderCompileGLSL. */
void mglShaderFree(void *bytes);

/* Compare the vertex/fragment shader interfaces: varying names, types
 * and interface blocks must match across stages.  On success returns 0;
 * on mismatch or a parse/sema failure returns -1 and writes a
 * NUL-terminated message into err_buf (err_cap bytes) if non-NULL. */
int mglShaderInterfaceCheck(const char *vs_src, const char *fs_src,
                            char *err_buf, size_t err_cap);

#ifdef __cplusplus
}
#endif

#endif /* MGL_SHADER_ABI_H */
