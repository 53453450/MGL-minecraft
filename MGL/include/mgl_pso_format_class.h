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
 * mgl_pso_format_class.h
 *
 * C1 / O3.2 domain strip from mgl_render.cpp — format-class PSO builder
 * policy (CTS Batch 4 / OBJC O3.2). Topology class, tessellation PSO
 * modes, color/depth/stencil format-class selection, blend/stencil/cull
 * GL→Metal maps, and pass scissor/viewport clamps. Pure classification
 * + maps; no Metal-cpp, no renderer instance.
 *
 * Do not sink these helpers back into mgl_render.cpp.
 * Do not grow +Binding.m / +RenderPass.m into thick shells — ObjC stays
 * a thin apply / materialize port.
 *
 * Callers historically went through mgl_render.h; that header includes
 * this one so RenderPass / PipelineCache keep call sites.
 */

#ifndef MGL_PSO_FORMAT_CLASS_H
#define MGL_PSO_FORMAT_CLASS_H

#include <stdint.h>

#include "glm_context.h"   /* Program */

#ifdef __cplusplus
extern "C" {
#endif

int mglRenderNeedsExplicitTopology(int geometry_expansion, uint32_t last_draw_mode,
                                   int vs_writes_layer);
uint32_t mglRenderPrimitiveTopologyClass(uint32_t gl_mode);
uint32_t mglRenderTessPartitionMode(uint32_t tess_gen_spacing);
uint32_t mglRenderTessOutputWinding(uint32_t tess_gen_vertex_order);
uint32_t mglRenderTessControlPointIndexType(int indexed_draw);
int mglRenderRasterizationEnabled(int rasterizer_discard, int has_fragment);
int mglRenderPipelineFunctionsReady(int has_vs, int has_fs, int rasterizer_discard);
/* Does the vertex stage write gl_Layer ([[render_target_array_index]])?  Read
 * from the exact per-stage builtin mask; the previous source-text scan matched
 * comments and longer identifiers too. */
int mglRenderVSWritesLayer(const Program *vertex_program);
uint32_t mglRenderDepthFormatOrFallback(uint32_t format);
uint32_t mglRenderStencilFormatOrFallback(uint32_t format);
int mglRenderColorAttachmentBitfieldDone(uint32_t bitfield, int index);
uint32_t mglRenderMaxTessellationFactor(void);
uint32_t mglRenderDefaultFBOStencilFormat(uint32_t format);
int mglRenderColor0IntentionallyDisabled(int has_fbo, uint32_t draw_buffer0);
int mglRenderColorFormatNeedsFallback(uint32_t format);
uint32_t mglRenderDefaultColorPixelFormat(void);
uint32_t mglRenderColorFormatOrBGRA(uint32_t format);
int mglRenderPipelineFormatCompatible(uint32_t cached, uint32_t built);
int mglRenderPipelinePassColorMismatch(uint32_t pipeline, uint32_t pass);
int mglRenderPipelinePassAttachmentMismatch(uint32_t pipeline, uint32_t pass);
int mglRenderSkipInvalidColorAttachment(uint32_t format);
int mglRenderPixelFormatIsInvalid(uint32_t format);
uint32_t mglRenderInvalidPixelFormat(void);
uint32_t mglRenderAttachmentFormatOrInvalid(int has_metal, uint32_t mapped);
int mglRenderPassUnifyPackedDS(uint32_t depth_format, uint32_t stencil_format,
                               uint32_t *out_format);
int mglRenderClearRectPipelineReady(int writes_color, uint32_t color_format,
                                    int writes_depth, uint32_t depth_format);
int mglRenderDrawBufferIsNone(uint32_t draw_buffer);
uint32_t mglRenderBlendingEnabledMaskBit(int blend_enabled, int index);
int mglRenderClearColorWriteMasks(int rasterizer_discard, int tess_capture,
                                  int cull_capture);
int mglRenderNeedsVertexDescriptor(int geometry_expansion, int tess_compute);
int mglRenderColorAttachmentBitSet(uint32_t bitfield, uint32_t index);
int mglRenderSampledRTNeedsCopy(int is_rt, uint32_t write_version);
int mglRenderSampledRTCopyStale(uint32_t sampled_version, uint32_t rt_version);
int mglRenderNativeAttribIndexValid(uint32_t index);
uint32_t mglRenderNativeAttribStepFunction(void);
int mglRenderSkipUnboundAttrib(int uses_current, int has_binding);
int mglRenderAttribFormatMapped(uint32_t format);
int mglRenderVertexBufferIndexValid(int index, uint32_t max);
void mglRenderAttribStepFromDivisor(int uses_current, uint32_t divisor,
                                    uint32_t *step_fn, uint32_t *step_rate);
uint32_t mglRenderAttribCountAfter(uint32_t current, uint32_t index);
int mglRenderApplyBlendRepair(int valid, uint32_t *value, uint32_t fallback);
uint32_t mglRenderColorWriteMaskFromChannels(int use_mask, int r, int g, int b,
                                            int a);
uint32_t mglRenderForceDefaultFBOAlphaWrite(int attachment, int has_fbo,
                                           uint32_t mask);
int mglRenderBlendFactorFromGL(uint32_t gl_blend, uint32_t *out);
int mglRenderBlendOperationFromGL(uint32_t gl_op, uint32_t *out);
int mglRenderStencilOpFromGL(uint32_t gl_op, uint32_t *out);
int mglRenderFrontFaceValid(uint32_t front_face);
uint32_t mglRenderFrontFaceOrCCW(uint32_t front_face);
int mglRenderSkipCullForSampledPass(int has_fbo, int depth_test, int fs_sampled,
                                    int rt_copy);
uint32_t mglRenderCullModeFromGL(int cull_enabled, uint32_t cull_face_mode);
uint32_t mglRenderDepthClipMode(int depth_clamp);
int mglRenderPolygonOffsetEnabled(int fill, int line, int point);
uint32_t mglRenderTriangleFillMode(uint32_t polygon_mode);
int mglRenderPolygonModeValid(uint32_t mode);
uint32_t mglRenderPolygonModeOrFill(uint32_t mode);
int mglRenderUseDepthState(int depth_test, int pass_has_depth);
int mglRenderUseStencilState(int stencil_test, int pass_has_stencil);
int mglRenderSuppressDepthStencilWrites(int rasterizer_discard, int tess_capture,
                                        int cull_capture);
uint32_t mglRenderStencilWriteMask(int suppress, uint32_t mask);
void mglRenderClampScissorRect(int32_t *x, int32_t *y, int32_t *w, int32_t *h,
                               uint32_t pass_w, uint32_t pass_h);
int32_t mglRenderMetalScissorY(int32_t y, int32_t h, uint32_t pass_h,
                               uint32_t clip_origin);
void mglRenderClampViewport(double *x, double *y, double *w, double *h,
                            uint32_t pass_w, uint32_t pass_h);
double mglRenderMetalViewportY(double y, double h, uint32_t pass_h);
uint32_t mglRenderCompareFuncOrFallback(uint32_t func, int valid,
                                        uint32_t fallback);
uint32_t mglRenderDepthWriteEnabled(int writemask, int suppress);
int mglRenderDrawModeFullyCulled(int cull_face, uint32_t cull_face_mode,
                                 int produces_polygons);

/* Format-class helpers required by DepthFormatOrFallback / PassUnifyPackedDS
 * (also used by later depth-blit residual in mgl_render.cpp). */
int mglRenderPixelFormatIsPackedDepthStencil(uint32_t pixel_format);
uint32_t mglRenderDefaultDepthPixelFormat(void);

#ifdef __cplusplus
}
#endif

#endif /* MGL_PSO_FORMAT_CLASS_H */
