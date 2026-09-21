/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef MGL_RENDER_API_DRAW_H
#define MGL_RENDER_API_DRAW_H

/* Declarations for the draw slice of the renderer facade.
 * Value layouts live in mgl_render.h. Standalone include pulls
 * mgl_render_fwd.h (incomplete types) instead of the full facade. */

#ifndef MGL_RENDER_H
#include "mgl_render_fwd.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

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
