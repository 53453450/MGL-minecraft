/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#include "mgl_draw_cull.h"

#include "mgl_draw_tess.h"
#include "mgl_render.h"
#include "mgl_shader_abi.h"

#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

extern "C" Program *mglResolveProgramForStageFromState(GLMContext ctx, int stage);

namespace {

static int cull_ops_ready(const MGLCullDistanceHostOps *ops)
{
    return ops && ops->renderer && ops->bind_mtl_program && ops->create_buffer &&
           ops->clear_cull_capture && ops->set_cull_capture_active &&
           ops->store_cull_capture && ops->mark_dirty_all &&
           ops->process_gl_state && ops->encoder_has_current &&
           ops->encoder_owner && ops->mark_cb_has_work &&
           ops->end_render_encoding;
}

static Program *cull_vs(GLMContext ctx)
{
    return ctx ? mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER) : nullptr;
}

} // namespace

extern "C" int mglDrawRunCullDistanceArrayCapture(
    GLMContext ctx, GLint first, GLsizei count, GLsizei instanceCount,
    GLuint baseInstance, const MGLCullDistanceHostOps *ops)
{
    if (!cull_ops_ready(ops) || !ctx || first < 0 || count <= 0 ||
        instanceCount <= 0) {
        if (ops && ops->clear_cull_capture) {
            ops->clear_cull_capture(ops->renderer);
        }
        return 0;
    }
    ops->clear_cull_capture(ops->renderer);

    Program *vertexProgram = cull_vs(ctx);
    if (!vertexProgram || !vertexProgram->uses_cull_distance ||
        !ops->bind_mtl_program(ops->renderer, vertexProgram) ||
        !vertexProgram->modules[_VERTEX_SHADER].mtl_cull_capture_function) {
        return 0;
    }

    uint64_t captureBytes = 0u;
    if (mglRenderCullDistanceCaptureBytes((uint32_t)first, (uint32_t)count,
                                          (uint32_t)instanceCount,
                                          &captureBytes) != 0) {
        return 0;
    }
    void *capture = ops->create_buffer(ops->renderer, captureBytes);
    if (!capture) {
        return 0;
    }
    if (ops->set_ctx) {
        ops->set_ctx(ops->renderer, ctx);
    }
    ops->set_cull_capture_active(ops->renderer, 1);
    ops->mark_dirty_all(ctx);
    if (!ops->process_gl_state(ops->renderer) ||
        ops->encoder_has_current(ops->renderer) != 1) {
        ops->set_cull_capture_active(ops->renderer, 0);
        ops->mark_dirty_all(ctx);
        /* create_buffer returns retained; drop on session failure. */
        CFRelease(capture);
        return 0;
    }

    MGLCullDistanceEmuParams params;
    mglRenderFillCullDistanceEmuParams(
        1u, (uint32_t)first, nullptr, 0u, 0u, 32u,
        (uint32_t)std::min<GLuint>(vertexProgram->cull_distance_count, 8u),
        baseInstance, (uint32_t)count, &params);
    mglRenderBindCullDistanceEmuSlots(ops->encoder_owner(ops->renderer), capture,
                                      &params);
    mglTessEncodeCaptureArray(ops->encoder_owner(ops->renderer), (uint32_t)first,
                              (uint32_t)count, (uint32_t)instanceCount,
                              baseInstance);
    ops->mark_cb_has_work(ops->renderer);
    ops->end_render_encoding(ops->renderer);
    ops->set_cull_capture_active(ops->renderer, 0);
    ops->store_cull_capture(ops->renderer, capture, baseInstance,
                            (uint32_t)count);
    ops->mark_dirty_all(ctx);
    return 1;
}

extern "C" int mglDrawRunCullDistanceElementCapture(
    GLMContext ctx, const uint8_t *indexBytes, GLenum indexType, GLsizei count,
    GLint baseVertex, GLsizei instanceCount, GLuint baseInstance,
    const MGLCullDistanceHostOps *ops)
{
    if (!ops || !ctx || !indexBytes || count <= 0 || instanceCount <= 0) {
        return 0;
    }
    uint32_t restartIndex = 0u;
    const int restartEnabled =
        ops->primitive_restart
            ? ops->primitive_restart(ctx, indexType, &restartIndex)
            : 0;
    const uint32_t elemWidth = mglRenderGLIndexElementSize((uint64_t)indexType);
    int32_t first = 0;
    uint32_t vertexCount = 0u;
    if (mglRenderPlanCullDistanceElementRange(
            indexBytes, elemWidth, (uint32_t)count, restartEnabled ? 1 : 0,
            restartIndex, baseVertex, &first, &vertexCount) != 0) {
        return 0;
    }
    return mglDrawRunCullDistanceArrayCapture(ctx, first, (GLsizei)vertexCount,
                                              instanceCount, baseInstance, ops);
}

extern "C" int mglDrawEncodeCullDistanceArray(
    GLMContext ctx, GLenum mode, GLint first, GLsizei count,
    GLsizei instanceCount, GLuint baseInstance, const void *enc_ctx,
    const MGLCullDistanceHostOps *ops)
{
    if (!ops || !ops->renderer || !ops->bind_cull_emu ||
        !ops->encode_context_active || !ops->encode_context_active(enc_ctx)) {
        return 0;
    }
    Program *active = cull_vs(ctx);
    if (!active || !active->uses_cull_distance) {
        return 0;
    }
    if (ops->try_array_split_encode && ops->device && ops->encoder_owner) {
        void *dev = ops->device(ops->renderer);
        void *enc = ops->encoder_owner(ops->renderer);
        if (ops->try_array_split_encode(ops->renderer, dev, enc, mode, first,
                                        count, (uint64_t)instanceCount,
                                        (uint64_t)baseInstance, enc_ctx)) {
            return 1;
        }
    }
    ops->bind_cull_emu(ops->renderer, mode, (GLuint)first, nullptr, 0u, enc_ctx);
    return 0;
}

extern "C" int mglDrawEncodeCullDistanceElement(
    GLMContext ctx, GLenum mode, const uint8_t *indexBytes, GLenum indexType,
    GLsizei count, GLint baseVertex, GLsizei instanceCount, GLuint baseInstance,
    int polygon_line_mode, const void *enc_ctx,
    const MGLCullDistanceHostOps *ops)
{
    if (!ops || !ops->renderer || !ops->bind_cull_emu ||
        !ops->draw_indexed_primitives || !ops->device || !ops->encoder_owner) {
        return 0;
    }
    Program *active = cull_vs(ctx);
    if (!active || !active->uses_cull_distance) {
        return 0;
    }
    void *captureBuffer =
        ops->load_cull_capture ? ops->load_cull_capture(ops->renderer) : nullptr;
    if (active->modules[_VERTEX_SHADER].mtl_cull_capture_function &&
        !captureBuffer) {
        return 1; /* handled as no-op (match ObjC early YES) */
    }
    if (!indexBytes || count <= 0 || instanceCount <= 0 ||
        !ops->encode_context_active || !ops->encode_context_active(enc_ctx)) {
        return 1;
    }

    uint32_t restartIndex = 0u;
    const int restartEnabled =
        ops->primitive_restart
            ? ops->primitive_restart(ctx, indexType, &restartIndex)
            : 0;
    void *planOwner = nullptr;
    void *indexBufferHandle = nullptr;
    uint64_t primitiveCount = 0u;
    if (mglRenderCreateCullDistanceIndexPlan(
            ops->device(ops->renderer), indexBytes, indexType, (uint64_t)count,
            mode, restartEnabled ? 1 : 0, restartIndex, baseVertex,
            polygon_line_mode ? 1 : 0, &planOwner, &indexBufferHandle,
            &primitiveCount) != 0 ||
        !planOwner) {
        return 1;
    }

    void *encoder = nullptr;
    if (enc_ctx) {
        /* C++ forbids defining a struct type inside a cast (C allowed the
         * anonymous-struct trick); use a named layout-compatible view of
         * the caller's encode context (first member: renderer owner). */
        struct CullEncodeCtxView {
            void *render_encoder_owner;
        };
        encoder = ((const CullEncodeCtxView *)enc_ctx)->render_encoder_owner;
    }
    if (!encoder) {
        encoder = ops->encoder_owner(ops->renderer);
    }

    for (uint64_t primitiveIndex = 0u; primitiveIndex < primitiveCount;
         ++primitiveIndex) {
        MGLRenderCullDistancePrimitive primitive = {};
        if (mglRenderGetCullDistanceIndexPrimitive(planOwner, primitiveIndex,
                                                   &primitive) != 0) {
            break;
        }
        ops->bind_cull_emu(ops->renderer, mode, 0u, primitive.vertices,
                           primitive.vertex_count, enc_ctx);
        ops->draw_indexed_primitives(
            encoder, primitive.primitive_type, primitive.index_count,
            indexBufferHandle, primitive.index_buffer_offset,
            (uint64_t)instanceCount, 0, (uint64_t)baseInstance);
    }
    mglRenderDestroyCullDistanceIndexPlan(&planOwner);
    return 1;
}

extern "C" int mglDrawPrepareAndEncodeCullDistanceElement(
    GLMContext ctx, GLenum mode, const uint8_t *indexBytes, GLenum indexType,
    GLsizei count, GLint baseVertex, GLsizei instanceCount, GLuint baseInstance,
    int polygon_line_mode, const MGLCullDistanceHostOps *ops)
{
    if (!ops || !ops->renderer) {
        return 0;
    }
    Program *active = cull_vs(ctx);
    if (!active || !active->uses_cull_distance) {
        return 0;
    }
    if (!indexBytes || count <= 0 || instanceCount <= 0) {
        return 1;
    }

    if (active->modules[_VERTEX_SHADER].mtl_cull_capture_function) {
        if (!mglDrawRunCullDistanceElementCapture(
                ctx, indexBytes, indexType, count, baseVertex, instanceCount,
                baseInstance, ops) ||
            !ops->process_gl_state(ops->renderer) ||
            ops->encoder_has_current(ops->renderer) != 1) {
            return 1;
        }
    }

    struct {
        void *render_encoder_owner;
    } encCtx = {ops->encoder_owner(ops->renderer)};
    return mglDrawEncodeCullDistanceElement(
        ctx, mode, indexBytes, indexType, count, baseVertex, instanceCount,
        baseInstance, polygon_line_mode, &encCtx, ops);
}
