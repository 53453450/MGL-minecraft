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
 * mgl_pso_format_class.c — C1 / O3.2 strip from mgl_render.cpp.
 * Format-class PSO builder: topology / tess modes / attachment format-class
 * / blend·stencil·cull maps / scissor·viewport clamps (CTS Batch 4).
 * Pure C; GL enums via glcorearb.h; Metal value ABI via mgl_render_values.h.
 * ObjC RenderPass / PipelineCache stay thin apply ports — do not grow them.
 */

#include "mgl_pso_format_class.h"

#include "glcorearb.h"
#include "mgl_render_values.h"

#include <stdint.h>
#include <string.h>

int mglRenderNeedsExplicitTopology(int geometry_expansion, uint32_t last_draw_mode,
                                   int vs_writes_layer) {
    return geometry_expansion || last_draw_mode == GL_POINTS || vs_writes_layer
               ? 1
               : 0;
}

uint32_t mglRenderPrimitiveTopologyClass(uint32_t gl_mode) {
    switch (gl_mode) {
    case GL_POINTS:
        return MGLPrimitiveTopologyClassPoint;
    case GL_LINES:
    case GL_LINE_STRIP:
    case GL_LINE_LOOP:
    case GL_LINES_ADJACENCY:
    case GL_LINE_STRIP_ADJACENCY:
        return MGLPrimitiveTopologyClassLine;
    default:
        return MGLPrimitiveTopologyClassTriangle;
    }
}

uint32_t mglRenderTessPartitionMode(uint32_t tess_gen_spacing) {
    switch (tess_gen_spacing) {
    case GL_FRACTIONAL_EVEN:
        return MGLTessellationPartitionModeFractionalEven;
    case GL_FRACTIONAL_ODD:
        return MGLTessellationPartitionModeFractionalOdd;
    default:
        return MGLTessellationPartitionModeInteger;
    }
}

uint32_t mglRenderTessOutputWinding(uint32_t tess_gen_vertex_order) {
    return tess_gen_vertex_order == GL_CW ? MGLWindingClockwise
                                          : MGLWindingCounterClockwise;
}

uint32_t mglRenderTessControlPointIndexType(int indexed_draw) {
    return indexed_draw ? MGLTessellationControlPointIndexTypeUInt32
                        : MGLTessellationControlPointIndexTypeNone;
}

int mglRenderRasterizationEnabled(int rasterizer_discard, int has_fragment) {
    return rasterizer_discard ? (has_fragment ? 1 : 0) : 1;
}

int mglRenderPipelineFunctionsReady(int has_vs, int has_fs, int rasterizer_discard) {
    return has_vs && (has_fs || rasterizer_discard) ? 1 : 0;
}

int mglRenderVSWritesLayer(const char *src) {
    return src && strstr(src, "gl_Layer") ? 1 : 0;
}

uint32_t mglRenderDepthFormatOrFallback(uint32_t format) {
    return format == 0u ? mglRenderDefaultDepthPixelFormat() : format;
}

uint32_t mglRenderStencilFormatOrFallback(uint32_t format) {
    return format == 0u ? 253u /* Stencil8 */ : format;
}

int mglRenderColorAttachmentBitfieldDone(uint32_t bitfield, int index) {
    return (bitfield >> (index + 1)) == 0u ? 1 : 0;
}

uint32_t mglRenderMaxTessellationFactor(void) {
    return 64u;
}

uint32_t mglRenderDefaultFBOStencilFormat(uint32_t format) {
    return format == 0u || format == 260u /* Depth32Float_Stencil8 */
               ? 253u /* Stencil8 */
               : format;
}

int mglRenderColor0IntentionallyDisabled(int has_fbo, uint32_t draw_buffer0) {
    return has_fbo && draw_buffer0 == GL_NONE ? 1 : 0;
}

int mglRenderColorFormatNeedsFallback(uint32_t format) {
    return format == 0u ? 1 : 0;
}

uint32_t mglRenderDefaultColorPixelFormat(void) {
    return 80u; /* BGRA8Unorm */
}

uint32_t mglRenderColorFormatOrBGRA(uint32_t format) {
    return format == 0u ? mglRenderDefaultColorPixelFormat() : format;
}

int mglRenderPipelineFormatCompatible(uint32_t cached, uint32_t built) {
    return cached == 0u || built == 0u || cached == built ? 1 : 0;
}

int mglRenderPipelinePassColorMismatch(uint32_t pipeline, uint32_t pass) {
    return !mglRenderPixelFormatIsInvalid(pipeline) &&
                   !mglRenderPixelFormatIsInvalid(pass) && pipeline != pass
               ? 1
               : 0;
}

int mglRenderPipelinePassAttachmentMismatch(uint32_t pipeline, uint32_t pass) {
    if (mglRenderPixelFormatIsInvalid(pipeline) &&
        mglRenderPixelFormatIsInvalid(pass)) {
        return 0;
    }
    return pipeline != pass ? 1 : 0;
}

int mglRenderSkipInvalidColorAttachment(uint32_t format) {
    return format == 0u ? 1 : 0;
}

int mglRenderPixelFormatIsInvalid(uint32_t format) {
    return format == 0u ? 1 : 0;
}

uint32_t mglRenderInvalidPixelFormat(void) {
    return 0u;
}

uint32_t mglRenderAttachmentFormatOrInvalid(int has_metal, uint32_t mapped) {
    return has_metal ? mapped : mglRenderInvalidPixelFormat();
}

int mglRenderPassUnifyPackedDS(uint32_t depth_format, uint32_t stencil_format,
                               uint32_t *out_format) {
    if (mglRenderPixelFormatIsInvalid(depth_format) ||
        mglRenderPixelFormatIsInvalid(stencil_format) ||
        depth_format == stencil_format) {
        return 0;
    }
    if (mglRenderPixelFormatIsPackedDepthStencil(stencil_format)) {
        if (out_format) {
            *out_format = stencil_format;
        }
        return 1;
    }
    if (mglRenderPixelFormatIsPackedDepthStencil(depth_format)) {
        if (out_format) {
            *out_format = depth_format;
        }
        return 1;
    }
    return 0;
}

int mglRenderClearRectPipelineReady(int writes_color, uint32_t color_format,
                                    int writes_depth, uint32_t depth_format) {
    if (!writes_color && !writes_depth) {
        return 0;
    }
    if (writes_color && mglRenderPixelFormatIsInvalid(color_format)) {
        return 0;
    }
    if (writes_depth && mglRenderPixelFormatIsInvalid(depth_format)) {
        return 0;
    }
    return 1;
}

int mglRenderDrawBufferIsNone(uint32_t draw_buffer) {
    return draw_buffer == GL_NONE ? 1 : 0;
}

uint32_t mglRenderBlendingEnabledMaskBit(int blend_enabled, int index) {
    return blend_enabled ? (1u << index) : 0u;
}

int mglRenderClearColorWriteMasks(int rasterizer_discard, int tess_capture,
                                  int cull_capture) {
    return rasterizer_discard || tess_capture || cull_capture ? 1 : 0;
}

int mglRenderNeedsVertexDescriptor(int geometry_expansion, int tess_compute) {
    /* tess_compute suppresses the vertex descriptor for BOTH the compute
     * expansion (records bound at slot 28) and the TES-vertex render path
     * (which takes no [[attribute]] inputs and binds its streams by slot). */
    return !(geometry_expansion || tess_compute) ? 1 : 0;
}

int mglRenderColorAttachmentBitSet(uint32_t bitfield, uint32_t index) {
    return ((bitfield >> index) & 1u) != 0u ? 1 : 0;
}

int mglRenderSampledRTNeedsCopy(int is_rt, uint32_t write_version) {
    return is_rt && write_version != 0u ? 1 : 0;
}

int mglRenderSampledRTCopyStale(uint32_t sampled_version, uint32_t rt_version) {
    return sampled_version != rt_version ? 1 : 0;
}

int mglRenderNativeAttribIndexValid(uint32_t index) {
    return index < 32u ? 1 : 0;
}

uint32_t mglRenderNativeAttribStepFunction(void) {
    return 4u; /* MTLVertexStepFunctionPerPatch */
}

int mglRenderSkipUnboundAttrib(int uses_current, int has_binding) {
    return !uses_current && !has_binding ? 1 : 0;
}

int mglRenderAttribFormatMapped(uint32_t format) {
    return format != 0u ? 1 : 0;
}

int mglRenderVertexBufferIndexValid(int index, uint32_t max) {
    return index >= 0 && (uint32_t)index < max ? 1 : 0;
}

void mglRenderAttribStepFromDivisor(int uses_current, uint32_t divisor,
                                    uint32_t *step_fn, uint32_t *step_rate) {
    if (!uses_current && divisor) {
        if (step_fn) {
            *step_fn = 2u; /* MTLVertexStepFunctionPerInstance */
        }
        if (step_rate) {
            *step_rate = divisor;
        }
        return;
    }
    if (step_fn) {
        *step_fn = 1u; /* MTLVertexStepFunctionPerVertex */
    }
    if (step_rate) {
        *step_rate = 1u;
    }
}

uint32_t mglRenderAttribCountAfter(uint32_t current, uint32_t index) {
    return index + 1u > current ? index + 1u : current;
}

int mglRenderApplyBlendRepair(int valid, uint32_t *value, uint32_t fallback) {
    if (valid || !value) {
        return 0;
    }
    *value = fallback;
    return 1;
}

uint32_t mglRenderColorWriteMaskFromChannels(int use_mask, int r, int g, int b,
                                            int a) {
    if (!use_mask) {
        return 15u;
    }
    uint32_t mask = 0u;
    if (r) {
        mask |= 1u;
    }
    if (g) {
        mask |= 2u;
    }
    if (b) {
        mask |= 4u;
    }
    if (a) {
        mask |= 8u;
    }
    return mask;
}

uint32_t mglRenderForceDefaultFBOAlphaWrite(int attachment, int has_fbo,
                                           uint32_t mask) {
    return attachment == 0 && !has_fbo ? (mask | 8u) : mask;
}

int mglRenderBlendFactorFromGL(uint32_t gl_blend, uint32_t *out) {
    uint32_t factor = MGLBlendFactorZero;
    int known = 1;
    switch (gl_blend) {
    case GL_ZERO:
        factor = MGLBlendFactorZero;
        break;
    case GL_ONE:
        factor = MGLBlendFactorOne;
        break;
    case GL_SRC_COLOR:
        factor = MGLBlendFactorSourceColor;
        break;
    case GL_ONE_MINUS_SRC_COLOR:
        factor = MGLBlendFactorOneMinusSourceColor;
        break;
    case GL_DST_COLOR:
        factor = MGLBlendFactorDestinationColor;
        break;
    case GL_ONE_MINUS_DST_COLOR:
        factor = MGLBlendFactorOneMinusDestinationColor;
        break;
    case GL_SRC_ALPHA:
        factor = MGLBlendFactorSourceAlpha;
        break;
    case GL_ONE_MINUS_SRC_ALPHA:
        factor = MGLBlendFactorOneMinusSourceAlpha;
        break;
    case GL_DST_ALPHA:
        factor = MGLBlendFactorDestinationAlpha;
        break;
    case GL_ONE_MINUS_DST_ALPHA:
        factor = MGLBlendFactorOneMinusDestinationAlpha;
        break;
    case GL_CONSTANT_COLOR:
        factor = MGLBlendFactorBlendColor;
        break;
    case GL_ONE_MINUS_CONSTANT_COLOR:
        factor = MGLBlendFactorOneMinusBlendColor;
        break;
    case GL_CONSTANT_ALPHA:
        factor = MGLBlendFactorBlendAlpha;
        break;
    case GL_ONE_MINUS_CONSTANT_ALPHA:
        factor = MGLBlendFactorOneMinusBlendAlpha;
        break;
    case GL_SRC_ALPHA_SATURATE:
        factor = MGLBlendFactorSourceAlphaSaturated;
        break;
    case GL_SRC1_COLOR:
        factor = MGLBlendFactorSource1Color;
        break;
    case GL_ONE_MINUS_SRC1_COLOR:
        factor = MGLBlendFactorOneMinusSource1Color;
        break;
    case GL_SRC1_ALPHA:
        factor = MGLBlendFactorSource1Alpha;
        break;
    case GL_ONE_MINUS_SRC1_ALPHA:
        factor = MGLBlendFactorOneMinusSource1Alpha;
        break;
    default:
        known = 0;
        factor = MGLBlendFactorZero;
        break;
    }
    if (out) {
        *out = factor;
    }
    return known;
}

int mglRenderBlendOperationFromGL(uint32_t gl_op, uint32_t *out) {
    uint32_t op = MGLBlendOperationAdd;
    int known = 1;
    switch (gl_op) {
    case GL_FUNC_ADD:
        op = MGLBlendOperationAdd;
        break;
    case GL_FUNC_SUBTRACT:
        op = MGLBlendOperationSubtract;
        break;
    case GL_FUNC_REVERSE_SUBTRACT:
        op = MGLBlendOperationReverseSubtract;
        break;
    case GL_MIN:
        op = MGLBlendOperationMin;
        break;
    case GL_MAX:
        op = MGLBlendOperationMax;
        break;
    default:
        known = 0;
        op = MGLBlendOperationAdd;
        break;
    }
    if (out) {
        *out = op;
    }
    return known;
}

int mglRenderStencilOpFromGL(uint32_t gl_op, uint32_t *out) {
    uint32_t op = 0u;
    int known = 1;
    switch (gl_op) {
    case GL_KEEP:
        op = 0u;
        break;
    case GL_ZERO:
        op = 1u;
        break;
    case GL_REPLACE:
        op = 2u;
        break;
    case GL_INCR:
        op = 3u;
        break;
    case GL_INCR_WRAP:
        op = 4u;
        break;
    case GL_DECR:
        op = 5u;
        break;
    case GL_DECR_WRAP:
        op = 6u;
        break;
    case GL_INVERT:
        op = 7u;
        break;
    default:
        known = 0;
        op = 0u;
        break;
    }
    if (out) {
        *out = op;
    }
    return known;
}

int mglRenderFrontFaceValid(uint32_t front_face) {
    return front_face == GL_CW || front_face == GL_CCW ? 1 : 0;
}

uint32_t mglRenderFrontFaceOrCCW(uint32_t front_face) {
    return mglRenderFrontFaceValid(front_face) ? front_face : (uint32_t)GL_CCW;
}

int mglRenderSkipCullForSampledPass(int has_fbo, int depth_test, int fs_sampled,
                                    int rt_copy) {
    return (!has_fbo && !depth_test && fs_sampled) || rt_copy ? 1 : 0;
}

uint32_t mglRenderCullModeFromGL(int cull_enabled, uint32_t cull_face_mode) {
    if (!cull_enabled) {
        return MGLCullModeNone;
    }
    switch (cull_face_mode) {
    case GL_BACK:
        return MGLCullModeBack;
    case GL_FRONT:
        return MGLCullModeFront;
    default:
        return MGLCullModeNone;
    }
}

uint32_t mglRenderDepthClipMode(int depth_clamp) {
    return depth_clamp ? MGLDepthClipModeClamp : MGLDepthClipModeClip;
}

int mglRenderPolygonOffsetEnabled(int fill, int line, int point) {
    return fill || line || point ? 1 : 0;
}

uint32_t mglRenderTriangleFillMode(uint32_t polygon_mode) {
    return polygon_mode == GL_LINE ? 1u : 0u;
}

int mglRenderPolygonModeValid(uint32_t mode) {
    return mode == GL_FILL || mode == GL_LINE || mode == GL_POINT ? 1 : 0;
}

uint32_t mglRenderPolygonModeOrFill(uint32_t mode) {
    return mglRenderPolygonModeValid(mode) ? mode : (uint32_t)GL_FILL;
}

int mglRenderUseDepthState(int depth_test, int pass_has_depth) {
    return depth_test && pass_has_depth ? 1 : 0;
}

int mglRenderUseStencilState(int stencil_test, int pass_has_stencil) {
    return stencil_test && pass_has_stencil ? 1 : 0;
}

int mglRenderSuppressDepthStencilWrites(int rasterizer_discard, int tess_capture,
                                        int cull_capture) {
    return rasterizer_discard || tess_capture || cull_capture ? 1 : 0;
}

uint32_t mglRenderStencilWriteMask(int suppress, uint32_t mask) {
    return suppress ? 0u : mask;
}

void mglRenderClampScissorRect(int32_t *x, int32_t *y, int32_t *w, int32_t *h,
                               uint32_t pass_w, uint32_t pass_h) {
    if (!x || !y || !w || !h) {
        return;
    }
    int32_t sx = *x;
    int32_t sy = *y;
    int32_t sw = *w;
    int32_t sh = *h;
    if (sx < 0) {
        sw += sx;
        sx = 0;
    }
    if (sy < 0) {
        sh += sy;
        sy = 0;
    }
    if (sx >= (int32_t)pass_w || sy >= (int32_t)pass_h) {
        *x = 0;
        *y = 0;
        *w = 0;
        *h = 0;
        return;
    }
    int32_t max_w = (int32_t)pass_w - sx;
    int32_t max_h = (int32_t)pass_h - sy;
    if (sw > max_w) {
        sw = max_w;
    }
    if (sh > max_h) {
        sh = max_h;
    }
    if (sw <= 0 || sh <= 0) {
        *x = 0;
        *y = 0;
        *w = 0;
        *h = 0;
        return;
    }
    *x = sx;
    *y = sy;
    *w = sw;
    *h = sh;
}

int32_t mglRenderMetalScissorY(int32_t y, int32_t h, uint32_t pass_h,
                               uint32_t clip_origin) {
    if (clip_origin == GL_UPPER_LEFT) {
        return y;
    }
    int32_t metal_y = (int32_t)pass_h - (y + h);
    return metal_y < 0 ? 0 : metal_y;
}

void mglRenderClampViewport(double *x, double *y, double *w, double *h,
                            uint32_t pass_w, uint32_t pass_h) {
    if (!x || !y || !w || !h) {
        return;
    }
    double vx = *x;
    double vy = *y;
    double vw = *w;
    double vh = *h;
    double pw = (double)pass_w;
    double ph = (double)pass_h;
    if (vw <= 0.0 || vh <= 0.0) {
        *x = 0.0;
        *y = 0.0;
        *w = pw;
        *h = ph;
        return;
    }
    if (vx < 0.0) {
        vw += vx;
        vx = 0.0;
    }
    if (vy < 0.0) {
        vh += vy;
        vy = 0.0;
    }
    if (vx >= pw || vy >= ph) {
        *x = 0.0;
        *y = 0.0;
        *w = pw;
        *h = ph;
        return;
    }
    double max_w = pw - vx;
    double max_h = ph - vy;
    if (vw > max_w) {
        vw = max_w;
    }
    if (vh > max_h) {
        vh = max_h;
    }
    if (vw <= 0.0 || vh <= 0.0) {
        *x = 0.0;
        *y = 0.0;
        *w = pw;
        *h = ph;
        return;
    }
    *x = vx;
    *y = vy;
    *w = vw;
    *h = vh;
}

double mglRenderMetalViewportY(double y, double h, uint32_t pass_h) {
    double metal_y = (double)pass_h - (y + h);
    return metal_y < 0.0 ? 0.0 : metal_y;
}

uint32_t mglRenderCompareFuncOrFallback(uint32_t func, int valid,
                                        uint32_t fallback) {
    return valid ? func : fallback;
}

uint32_t mglRenderDepthWriteEnabled(int writemask, int suppress) {
    return (!suppress && writemask) ? 1u : 0u;
}

int mglRenderDrawModeFullyCulled(int cull_face, uint32_t cull_face_mode,
                                 int produces_polygons) {
    return cull_face && cull_face_mode == GL_FRONT_AND_BACK && produces_polygons
               ? 1
               : 0;
}

int mglRenderPixelFormatIsPackedDepthStencil(uint32_t pixel_format) {
    return pixel_format == 255u /* Depth24Unorm_Stencil8 */ ||
                   pixel_format == 260u /* Depth32Float_Stencil8 */
               ? 1
               : 0;
}

uint32_t mglRenderDefaultDepthPixelFormat(void) {
    return 252u; /* Depth32Float */
}

