/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef MGL_RENDER_API_DRAW_H
#define MGL_RENDER_API_DRAW_H

/* Declarations for the draw slice of the renderer facade.
 * Value layouts are defined here; types of other subsystems that are only
 * passed by pointer are incomplete (mgl_render_fwd.h). */

#include "mgl_render_fwd.h"

#ifdef __cplusplus
extern "C" {
#endif

/* C1: IntegerReadback + Y-flip/depth/GetTexImagePlan/MSAA stride -> mgl_readback_policy.h */






/* TES XFB field byte size for a GL type (FLOAT/INT/
 * UINT + vec2/3/4; 0 for unsupported).  Matches mglTESXFBFieldByteSize and
 * the packed-write stride contract in mglFixMSLTesAsComputeKernel.  Shared
 * by both gates. */
uint64_t mglRenderTESXFBFieldByteSize(uint64_t gl_type);

/* overflow-checked product (a * b) for tessellation
 * size math; matches the ObjC mglCheckedNSUIntegerProduct.  Returns 0 with
 * *result set, -1 on bad args / overflow.  Shared by both gates. */
int mglRenderCheckedProduct(uint64_t a, uint64_t b, uint64_t *result);

/* TES XFB compact vertex stride — sum of the byte
 * sizes of the transform-feedback varyings resolved by name against the
 * TES stage-output resource list (lockstep with the packed writes injected
 * by mglFixMSLTesAsComputeKernel).  0 when the stride cannot be proven
 * (no varyings / unknown field type / overflow).  Matches the ObjC
 * mglTESXFBVertexStride.  Shared by both gates. */
uint64_t mglRenderTESXFBVertexStride(const void *program);

/* native TES interface support decision — module /
 * function presence, point-mode / XFB exclusion, TRI/QUADS gen-mode gate,
 * and the MTL::Function patchType + patchControlPointCount consistency
 * checks (zero control-point count = legacy encoding, tolerated).  Shared
 * by both gates; the ObjC caller passes __bridge'd MTL::Function pointers. */
int mglRenderNativeTESInterfaceSupported(
    void *tes_function,
    uint64_t tes_metallib_bytes,
    uint32_t tes_gen_point_mode,
    uint32_t tes_xfb_varying_count,
    uint32_t tes_gen_mode,
    void *tcs_function,
    uint64_t tcs_metallib_bytes,
    uint32_t tcs_output_vertices);

/* pure viewport/scissor/framebuffer intersection
 * decision for the per-draw rasterization-empty early-out.  Returns 1 when
 * the draw cannot rasterize any pixel, 0 otherwise (a zero pass size is
 * "not empty" — the caller resolves the pass size first).  Shared by both
 * gates. */
int mglRenderRasterizationIsEmpty(
    int32_t vx,
    int32_t vy,
    int32_t vw,
    int32_t vh,
    uint32_t pass_width,
    uint32_t pass_height,
    int32_t scissor_enabled,
    int32_t sx,
    int32_t sy,
    int32_t sw,
    int32_t sh);

/* quad-array emulation — for each group of 4 array
 * vertices emit `(a,a+1,a+2,a,a+2,a+3)` (two triangles), quad_count*6
 * uint32 total.  Pure CPU; caller frees.  Returns 0 with *out_count, -1 on
 * bad args. */
int mglRenderExpandQuadArrayIndices(
    uint32_t quad_count, uint32_t **out_indices, uint64_t *out_count);

int mglRenderClipOriginIsLowerLeft(uint32_t origin);

int mglRenderEmulateTriangleFan(uint32_t mode, int polygon_point);

int mglRenderEmulateLineLoop(uint32_t mode);

int mglRenderEmulateQuads(uint32_t mode, int polygon_point);

/* GL 4.6 §10.1.11–14: adjacency modes without an active GS ignore
 * adjacent vertices and emit the core line/triangle stream. */
int mglRenderEmulateAdjacency(uint32_t mode);

int mglRenderQuadsCountTooSmall(uint32_t mode, int32_t count);

int mglRenderPolygonPointEmulateMode(uint32_t mode);

int mglRenderCompareFuncFromGL(uint32_t func, uint32_t *out);

int mglRenderFrontFaceIsClockwise(uint32_t front_face);

int mglRenderFrontFaceIsCounterClockwise(uint32_t front_face);

/* FNV-1a single hash step; matches mglHashStepU64. */
/* Fixed restart-index for a type; matches the fixed branch of
 * mglPrimitiveRestartIndexForType.  1 if defined; *out set. */
/* GL uniform/attrib type -> element byte size; matches mglGLTypeElementByteSize. */

typedef struct MGLRenderGeometryGatherResult_t {
    uint32_t *gather;          /* malloc'd raw gather (vertex_ids) */
    uint32_t gather_count;
    uint32_t primitive_count;
    uint32_t max_index;
} MGLRenderGeometryGatherResult;

/* the indexed-PATCHES geometry gather — expand a raw
 * index stream (BYTE/SHORT/INT element size) into a flat vertex-id gather,
 * counting complete primitives of `last` vertices and dropping primitive
 * restarts / trailing incomplete groups.  Pure CPU; caller frees
 * result.gather.  Returns 0 on success, -1 on bad args / no valid gather. */
int mglRenderGeometryGatherIndices(
    const uint8_t *index_bytes,
    uint32_t index_type_byte_width,   /* 1, 2 or 4 */
    uint32_t count,
    int restart_enabled,
    uint32_t restart_index,
    uint32_t input_vertices,
    MGLRenderGeometryGatherResult *out);

typedef struct MGLRenderPolygonOffsetDecision_t {
    int triangle_fill_mode;      /* 0 = fill, 1 = lines */
    int needs_polygon_mode_repair;
    int enable_depth_bias;
} MGLRenderPolygonOffsetDecision;

/* polygon-offset draw decision — the triangle fill
 * mode (GL_LINE -> lines), the invalid polygon-mode repair condition and
 * the depth-bias enablement per polygon mode with the three capability
 * flags.  Pure decision shared by both gates. */
int mglRenderPolygonOffsetDecision(
    uint32_t mode,
    int has_ctx,
    int produces_polygons,
    uint32_t polygon_mode,
    int cap_point,
    int cap_line,
    int cap_fill,
    MGLRenderPolygonOffsetDecision *out);

/* Unified value-state draw plan. Resources are borrowed and final draw
 * encoding is owned by the C++ backend. */
typedef struct MGLRenderDrawPlan_t {
    uint32_t kind;              /* MGL_RENDER_DRAW_* */
    uint32_t primitive_type;    /* MTLPrimitiveType ABI value. */
    /* ARRAY: */
    uint64_t vertex_start;
    uint64_t vertex_count;
    /* INDEXED: */
    uint64_t index_count;
    uint32_t index_type;        /* MTLIndexType ABI value. */
    void *index_buffer;         /* +0 borrowed MTL::Buffer* */
    uint64_t index_buffer_offset;
    int64_t base_vertex;
    /* INDIRECT: */
    void *indirect_buffer;      /* +0 borrowed MTL::Buffer* */
    uint64_t indirect_buffer_offset;
    /* PATCHES（native TES）: */
    uint64_t control_point_count;
    uint64_t patch_start;
    uint64_t patch_count;
    void *patch_index_buffer;           /* +0 borrowed */
    uint64_t patch_index_buffer_offset;
    void *control_point_index_buffer;   /* +0 borrowed */
    uint64_t control_point_index_buffer_offset;
    /* Common fields. */
    uint64_t instance_count;
    uint64_t base_instance;
} MGLRenderDrawPlan;

enum {
    MGL_RENDER_DRAW_ARRAY = 1,
    MGL_RENDER_DRAW_INDEXED = 2,
    MGL_RENDER_DRAW_ARRAY_INDIRECT = 3,
    MGL_RENDER_DRAW_INDEXED_INDIRECT = 4,
    MGL_RENDER_DRAW_PATCHES = 5,
    MGL_RENDER_DRAW_INDEXED_PATCHES = 6,
};

typedef struct MGLRenderCullDistancePrimitive_t {
    uint32_t vertices[4];
    uint32_t vertex_count;
    uint32_t primitive_type;
    uint32_t index_count;
    uint64_t index_buffer_offset;
} MGLRenderCullDistancePrimitive;

/* Build a UInt32 index buffer whose records each represent one complete GL
 * primitive. The opaque owner retains the borrowed index buffer and the
 * per-primitive explicit vertex IDs used by exact gl_CullDistance emulation. */
int mglRenderCreateCullDistanceIndexPlan(void *device, const void *source_indices, uint32_t source_index_type, uint64_t source_index_count, uint32_t draw_mode, int primitive_restart_enabled, uint32_t primitive_restart_index, int64_t base_vertex, int polygon_line_mode, MGLCullDistanceIndexPlan **owner_out, void **index_buffer_out, uint64_t *primitive_count_out);

typedef struct MGLCullDistanceEmuParams_t {
    uint32_t prim_vertex_count;
    uint32_t culldist_offset;
    uint32_t vertex_stride;
    uint32_t culldist_size;
    uint32_t first_vertex;
    uint32_t explicit_vertex_count;
    uint32_t explicit_vertices[4];
    uint32_t first_instance;
    uint32_t instance_stride;
} MGLCullDistanceEmuParams;

typedef struct MGLRenderCullDistanceLayout_t {
    void *mtl_buffer;
    int64_t binding_offset;
    uint32_t stride;
    int64_t first_relative_offset;
    uint32_t culldist_size;
} MGLRenderCullDistanceLayout;

/* O1.3: ObjC fills VAO-resolved ports; C++ builds layout (+ optional dummy). */
typedef struct MGLRenderCullDistanceAttribPort {
    void *mtl_buffer; /* NULL / invalid → skipped */
    int64_t binding_offset;
    uint32_t stride;
    int64_t relativeoffset;
    uint8_t valid;
} MGLRenderCullDistanceAttribPort;


int mglRenderRepackTessFactorTriangles(
    const void *src,
    uint64_t src_bytes,
    void *dst,
    uint64_t dst_bytes,
    uint32_t patch_count);

uint64_t mglRenderTessPrimitiveCount(
    const void *factors,
    uint64_t bytes,
    uint32_t patch_count,
    uint32_t tess_gen_mode,
    uint32_t instance_count);

/* GL 4.6 section 11.2.2 (Tessellation Primitive Generation) patch discard
 * predicate.
 * Tests the applicable outer tessellation levels before any clamp to
 * one; non-positive or NaN outer levels discard the patch. NULL inputs are
 * conservatively treated as discarded.  Shared by both gates. */
bool mglRenderTessFactorsDiscardPatch(
    uint32_t gen_mode,
    const float *edge,
    const float *inside);

/* per-patch expanded item count for the isolines /
 * point-mode / XFB TES kernel — lockstep with mglTessDomainVertexCount
 * (mgl_tess_domain.h).  Returns 0 when discarded. */
uint32_t mglRenderTessEvalItemsPerPatch(
    const void *factor_record,
    uint32_t gen_mode,
    uint32_t spacing,
    uint32_t point_mode);

/* Seed TES output-record position.xyz with domain coordinates before AIR runs.
 * factor_record is the full canonical record including exact float levels. */
uint32_t mglRenderSeedTessDomain(const void *factor_record,
    uint32_t gen_mode, uint32_t spacing, uint32_t point_mode, uint32_t winding,
    void *records, uint32_t count, uint32_t stride);

/* GL 4.6 §11.2.2.2 subdivision-count rounding —
 * fractional_even -> next even (min 2), fractional_odd -> next odd,
 * otherwise ceil(level). Delegates to the pure tessellation domain layer. */
uint32_t mglRenderTessRoundLevelForSpacing(
    uint32_t spacing,
    uint32_t ceil_level);

/* GL type -> MTLVertexFormat ABI value for TES
 * control-point stage inputs (Float/Float2/3/4, Int/Int2/3/4,
 * UInt/UInt2/3/4, else 0 = MTLVertexFormatInvalid).  Values match the
 * macOS SDK enum (Float=28 ... UInt4=39).  Shared by both gates. */
uint32_t mglRenderTessControlPointFormat(uint64_t gl_type);

/* Vertex format of one location of a control-point input: a matrix type
 * contributes one column (`rows` floats), everything else maps exactly like
 * mglRenderTessControlPointFormat.  The native post-tessellation descriptor
 * declares one attribute per location, so a member spanning several
 * locations (array element / matrix column) is described by this per-location
 * format rather than by an aggregate type Metal does not accept. */
uint32_t mglRenderTessControlPointLocationFormat(uint64_t gl_type);

/* Overflow-checked tess capture size (records x stride, min_stride floor).
 * Returns 0 with size_out/offset_out set, -1 on bad args / overflow. */
int mglRenderCheckedTessCaptureSize(
    int64_t count,
    int64_t instance_count,
    uint64_t stride,
    uint64_t min_stride,
    uint64_t *size_out,
    uint64_t *offset_out);

/* GL draw mode -> MTLPrimitiveType numbering
 * (0=Point, 1=Line, 2=LineStrip, 3=Triangle, 4=TriangleStrip;
 * 0xFFFFFFFF for modes the renderer routes elsewhere).  Pure table shared
 * by both gates; the caller casts to MTLPrimitiveType. */
uint32_t mglRenderMTLPrimitiveTypeForGLMode(uint32_t mode);

/* GL element index type -> MTLIndexType numbering
 * (0=UInt16, 1=UInt32; 0xFFFFFFFF otherwise).  Pure table shared by both
 * gates; the caller casts to MTLIndexType. */
uint32_t mglRenderMTLIndexTypeForGLType(uint32_t gl_type);

/* O3.3: IntegerAttribDstIsInt..BindingOffsetInMetal -> mgl_binding_stage.h */
int mglRenderIndexStreamFits(uint64_t offset, uint64_t count, uint32_t elem_bytes,
                             uint64_t metal_len);

int mglRenderDrawModeNeedsEmulate(uint32_t mode);

int mglRenderIndexTypeIsU8(uint32_t type);

int mglRenderDrawModeIsTriangles(uint32_t mode);

int mglRenderDrawModeIsTriangleStrip(uint32_t mode);

int mglRenderIndexTypeIsU16(uint32_t type);

int mglRenderIndexTypeIsU32(uint32_t type);

int mglRenderShaderResourceIndexValid(int spvc_type, uint32_t index,
                                      uint32_t count);

/* GL draw mode -> primitive vertex count (for the
 * cull-distance emulation params; 1 for unknown modes).  Pure table shared
 * by both gates. */
uint32_t mglRenderPrimitiveVertexCountForMode(uint32_t mode);

/* Render draw command facade. Enum values are passed as uint32_t so the C ABI
 * remains independent of Metal headers. Resources are borrowed for encoding. */
int mglRenderDrawPrimitives(void *render_encoder,
                               uint32_t primitive_type,
                               uint64_t vertex_start,
                               uint64_t vertex_count,
                               uint64_t instance_count,
                               uint64_t base_instance);

int mglRenderDrawIndexedPrimitives(void *render_encoder,
                                      uint32_t primitive_type,
                                      uint64_t index_count,
                                      uint32_t index_type,
                                      void *index_buffer,
                                      uint64_t index_buffer_offset,
                                      uint64_t instance_count,
                                      int64_t base_vertex,
                                      uint64_t base_instance);

int mglRenderDrawPrimitivesIndirect(void *render_encoder,
                                       uint32_t primitive_type,
                                       void *indirect_buffer,
                                       uint64_t indirect_buffer_offset);

int mglRenderDrawIndexedPrimitivesIndirect(
    void *render_encoder,
    uint32_t primitive_type,
    uint32_t index_type,
    void *index_buffer,
    uint64_t index_buffer_offset,
    void *indirect_buffer,
    uint64_t indirect_buffer_offset);

int mglRenderGetCullDistanceIndexPrimitive(MGLCullDistanceIndexPlan *owner, uint64_t primitive_index, MGLRenderCullDistancePrimitive *primitive_out);

void mglRenderDestroyCullDistanceIndexPlan(MGLCullDistanceIndexPlan **owner);

/* 1 = mode is not a per-primitive array split. 0 = filled. -1 = overflow. */
int mglRenderFillCullDistanceArrayPrimitives(
    uint32_t draw_mode, int32_t first, uint64_t count,
    MGLRenderCullDistancePrimitive *out, uint32_t cap, uint32_t *out_count);

/* Same owner as the indexed plan. 1 = not a split mode. 0 = plan ready
 * (index buffer may be NULL for LINE_STRIP). -1 = allocation failure. */
int mglRenderCreateCullDistanceArrayPlan(void *device, uint32_t draw_mode, int32_t first, uint64_t count, MGLCullDistanceIndexPlan **owner_out, void **index_buffer_out, uint64_t *primitive_count_out);

uint32_t mglRenderCollectCullDistanceAttribs(const Program *program,
                                             uint32_t *out, uint32_t cap);

void mglRenderBuildCullDistanceLayoutFromPorts(
    MGLRenderCullDistanceLayout *layout,
    const MGLRenderCullDistanceAttribPort *ports, uint32_t port_count,
    void *dummy_mtl_buffer);

uint32_t mglRenderCullDistanceLayoutOffset(
    const MGLRenderCullDistanceLayout *layout);

void mglRenderFillCullDistanceEmuParams(
    uint32_t prim_vertex_count, uint32_t first_vertex,
    const uint32_t *explicit_vertices, uint32_t explicit_vertex_count,
    uint32_t culldist_offset, uint32_t vertex_stride, uint32_t culldist_size,
    uint32_t first_instance, uint32_t instance_stride,
    MGLCullDistanceEmuParams *out);

int mglRenderCullDistanceCaptureBytes(uint32_t first, uint32_t count,
                                      uint32_t instance_count,
                                      uint64_t *out_bytes);

int mglRenderDrawPatches(void *render_encoder,
                            uint64_t control_point_count,
                            uint64_t patch_start,
                            uint64_t patch_count,
                            void *patch_index_buffer,
                            uint64_t patch_index_buffer_offset,
                            uint64_t instance_count,
                            uint64_t base_instance);

int mglRenderDrawIndexedPatches(void *render_encoder,
                                   uint64_t control_point_count,
                                   uint64_t patch_start,
                                   uint64_t patch_count,
                                   void *patch_index_buffer,
                                   uint64_t patch_index_buffer_offset,
                                   void *control_point_index_buffer,
                                   uint64_t control_point_index_buffer_offset,
                                   uint64_t instance_count,
                                   uint64_t base_instance);

int mglRenderSetIndirectDrawIndexed(void *indirect_command,
                                       uint32_t primitive_type,
                                       uint64_t index_count,
                                       uint32_t index_type,
                                       void *index_buffer,
                                       uint64_t index_buffer_offset,
                                       uint64_t instance_count,
                                       int64_t base_vertex,
                                       uint64_t base_instance);

int mglRenderSetIndirectDraw(void *indirect_command,
                                uint32_t primitive_type,
                                uint64_t vertex_start,
                                uint64_t vertex_count,
                                uint64_t instance_count,
                                uint64_t base_instance);

#ifdef __cplusplus
}
#endif

#endif
