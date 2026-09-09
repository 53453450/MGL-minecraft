/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#include "mgl_draw_issue.h"

#include "mgl_draw_encode.h"
#include "mgl_draw_mode.h"
#include "mgl_draw_tess.h"
#include "mgl_frame_activity.h"
#include "mgl_index_buffer.h"
#include "mgl_trace_log.h"
#include "mgl_types_buffer.h"
#include "mgl_render.h"

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdarg>

extern "C" Program *mglResolveProgramForStageFromState(GLMContext ctx,
                                                       int stage);

static uint64_t mglIssueNonNegativeCount(GLsizei count, GLsizei instanceCount)
{
    const uint64_t n = count > 0 ? (uint64_t)count : 0u;
    const uint64_t inst = instanceCount > 0 ? (uint64_t)instanceCount : 0u;
    return n * inst;
}

static bool mglIssueTessIfNeeded(GLMContext ctx, void *renderer, GLenum *mode,
                                 GLint first, GLsizei count, GLenum indexType,
                                 const void *indices, GLint baseVertex,
                                 GLsizei instanceCount, GLuint baseInstance,
                                 const char *label)
{
    if (!mode) {
        return false;
    }
    Program *tcs =
        mglResolveProgramForStageFromState(ctx, _TESS_CONTROL_SHADER);
    Program *tes =
        mglResolveProgramForStageFromState(ctx, _TESS_EVALUATION_SHADER);
    const MGLTessDrawClass tessClass = mglTessClassifyDraw(
        ctx, *mode, count, instanceCount, tcs, tes, label);
    if (tessClass == MGL_TESS_DRAW_NOT_APPLICABLE) {
        return false;
    }
    if (tessClass != MGL_TESS_DRAW_ACTIVE) {
        return true;
    }
    return mglDrawHostHandleTessellation(renderer, ctx, mode, first, count,
                                         indexType, indices, baseVertex,
                                         instanceCount, baseInstance, label);
}

static bool mglIssueGsIfNeeded(GLMContext ctx, void *renderer, GLenum mode,
                               GLint first, GLsizei count, GLenum indexType,
                               const void *indices, GLint baseVertex,
                               GLsizei instanceCount, GLuint baseInstance,
                               const char *label)
{
    Program *gs = mglResolveProgramForStageFromState(ctx, _GEOMETRY_SHADER);
    if (!gs || !gs->shader_slots[_GEOMETRY_SHADER]) {
        return false;
    }
    return mglDrawHostHandleGeometry(renderer, ctx, mode, first, count,
                                     indexType, indices, baseVertex,
                                     instanceCount, baseInstance, label);
}

extern "C" void mglIssueDrawArrays(GLMContext ctx, void *renderer, GLenum mode,
                                   GLint first, GLsizei count,
                                   GLsizei instanceCount, GLuint baseInstance,
                                   const char *label)
{
    if (!ctx || !renderer || count <= 0 || instanceCount <= 0) {
        return;
    }

    mglDrawHostBindContext(renderer, ctx);
    mglDrawHostSetLastPrimitiveMode(renderer, mode);

    GLenum liveMode = mode;
    if (mglIssueTessIfNeeded(ctx, renderer, &liveMode, first, count, 0,
                             NULL, 0, instanceCount, baseInstance, label)) {
        return;
    }
    if (mglIssueGsIfNeeded(ctx, renderer, liveMode, first, count, 0, NULL,
                           0, instanceCount, baseInstance, label)) {
        return;
    }
    if (mglDrawHostHandleXFB(renderer, ctx, liveMode, first, count,
                             instanceCount, baseInstance)) {
        return;
    }

    (void)mglDrawHostCaptureCullDistanceArray(renderer, ctx, first, count,
                                              instanceCount, baseInstance);

    if (!mglDrawHostProcessGLStateLocked(renderer, true)) {
        MGL_FRAME_INC(g_mglDrawArraysSkippedSinceSwap);
        return;
    }
    if (mglDrawHostRasterizationIsEmpty(renderer) ||
        mglDrawHostModeFullyCulled(renderer, liveMode)) {
        return;
    }
    mglDrawHostApplyPolygonOffset(renderer, liveMode);
    if (!mglDrawHostEnsureRasterEncoder(renderer)) {
        MGL_FRAME_INC(g_mglDrawArraysSkippedSinceSwap);
        return;
    }
    if (!mglDrawHostValidateArrayVertexInputs(renderer, ctx, liveMode, first,
                                              count)) {
        MGL_FRAME_INC(g_mglDrawArraysSkippedSinceSwap);
        return;
    }

    if (mglDrawHostEncodeCullDistanceArray(renderer, liveMode, first, count,
                                           instanceCount, baseInstance)) {
        mglDrawHostRecordArraySubmitted(renderer, liveMode,
                                        mglIssueNonNegativeCount(count,
                                                                 instanceCount));
        mglDrawHostWatchdogArrays(renderer, ctx);
        return;
    }

    if (!mglEncodeDrawArraysForRenderEncoderOwner(
            mglDrawHostEncoderOwner(renderer), ctx, mglDrawHostDevice(renderer),
            liveMode, first, count, (size_t)instanceCount,
            (size_t)baseInstance, label)) {
        MGL_FRAME_INC(g_mglDrawArraysSkippedSinceSwap);
        return;
    }
    mglDrawHostRecordArraySubmitted(renderer, liveMode,
                                    mglIssueNonNegativeCount(count,
                                                             instanceCount));
    mglDrawHostWatchdogArrays(renderer, ctx);
}

extern "C" void mglIssueDrawElements(GLMContext ctx, void *renderer, GLenum mode,
                                     GLsizei count, GLenum type,
                                     const void *indices, GLsizei instanceCount,
                                     GLint baseVertex, GLuint baseInstance,
                                     const char *label)
{
    if (!ctx || !renderer || count <= 0 || instanceCount <= 0) {
        return;
    }

    mglDrawHostBindContext(renderer, ctx);
    mglDrawHostSetLastPrimitiveMode(renderer, mode);

    GLenum liveMode = mode;
    if (mglIssueTessIfNeeded(ctx, renderer, &liveMode, 0, count, type,
                             indices, baseVertex, instanceCount, baseInstance,
                             label)) {
        return;
    }
    if (mglIssueGsIfNeeded(ctx, renderer, liveMode, 0, count, type, indices,
                           baseVertex, instanceCount, baseInstance, label)) {
        return;
    }

    if (!mglDrawHostProcessGLStateLocked(renderer, true)) {
        MGL_FRAME_INC(g_mglDrawElementsSkippedSinceSwap);
        return;
    }
    if (mglDrawHostRasterizationIsEmpty(renderer) ||
        mglDrawHostModeFullyCulled(renderer, liveMode)) {
        return;
    }
    mglDrawHostApplyPolygonOffset(renderer, liveMode);
    if (!mglDrawHostEnsureRasterEncoder(renderer)) {
        MGL_FRAME_INC(g_mglDrawElementsSkippedSinceSwap);
        return;
    }

    Buffer *glBuffer = NULL;
    void *metalBuffer = NULL;
    if (!mglDrawHostResolveElementBuffer(renderer, ctx, label, &glBuffer,
                                         &metalBuffer)) {
        return;
    }

    if (mglDrawHostEncodeCullDistanceElements(renderer, liveMode, type, indices,
                                              count, baseVertex, instanceCount,
                                              baseInstance)) {
        mglDrawHostRecordElementSubmitted(
            renderer, liveMode,
            mglIssueNonNegativeCount(count, instanceCount));
        mglDrawHostWatchdogElements(renderer, ctx);
        return;
    }

    const size_t indexOffset = (size_t)(uintptr_t)indices;
    if (!mglEncodeDrawElementsForRenderEncoderOwner(
            mglDrawHostEncoderOwner(renderer), ctx, mglDrawHostDevice(renderer),
            glBuffer, metalBuffer, liveMode, type, indexOffset, count,
            (size_t)instanceCount, (int64_t)baseVertex, (size_t)baseInstance,
            label)) {
        MGL_FRAME_INC(g_mglDrawElementsSkippedSinceSwap);
        return;
    }
    mglDrawHostRecordElementSubmitted(renderer, liveMode,
                                      mglIssueNonNegativeCount(count,
                                                               instanceCount));
    mglDrawHostWatchdogElements(renderer, ctx);
}

extern "C" void mglIssueMultiDrawArrays(GLMContext ctx, void *renderer,
                                        GLenum mode, const GLint *first,
                                        const GLsizei *count, GLsizei drawcount,
                                        const char *label)
{
    if (!ctx || !renderer || !first || !count || drawcount <= 0) {
        return;
    }
    for (GLsizei i = 0; i < drawcount; ++i) {
        mglIssueDrawArrays(ctx, renderer, mode, first[i], count[i], 1, 0u,
                           label);
    }
}

extern "C" void mglIssueMultiDrawElements(GLMContext ctx, void *renderer,
                                          GLenum mode, const GLsizei *count,
                                          GLenum type,
                                          const void *const *indices,
                                          GLsizei drawcount,
                                          const GLint *basevertex,
                                          const char *label)
{
    if (!ctx || !renderer || !count || drawcount <= 0) {
        return;
    }
    for (GLsizei i = 0; i < drawcount; ++i) {
        const void *indexPtr = indices ? indices[i] : NULL;
        const GLint baseVertex = basevertex ? basevertex[i] : 0;
        mglIssueDrawElements(ctx, renderer, mode, count[i], type, indexPtr, 1,
                             baseVertex, 0u, label);
    }
}

static unsigned mglIssueProgramName(GLMContext ctx)
{
    return (ctx && ctx->active_state) ? (unsigned)ctx->active_state->program_name
                                      : 0u;
}

static bool mglIssueIndirectNeedsCPUExpand(GLMContext ctx, GLenum mode)
{
    return mode == GL_PATCHES || mode == GL_QUADS || mode == GL_LINE_LOOP ||
           mglDrawHostHasGeometry(ctx) || mglDrawHostUsesCullDistance(ctx);
}

static bool mglIssueIndirectPreamble(GLMContext ctx, void *renderer, GLenum mode,
                                     const char *label)
{
    if (!ctx || !renderer) {
        return false;
    }
    mglDrawHostBindContext(renderer, ctx);
    mglDrawHostSetLastPrimitiveMode(renderer, mode);
    if (!mglDrawHostProcessGLStateLocked(renderer, true)) {
        mglTraceLog("%s skip process_gl_state program=%u",
                    label ? label : "indirect", mglIssueProgramName(ctx));
        return false;
    }
    if (mglDrawHostRasterizationIsEmpty(renderer) ||
        mglDrawHostModeFullyCulled(renderer, mode)) {
        mglTraceLog("%s skip empty_or_culled mode=0x%x program=%u",
                    label ? label : "indirect", (unsigned)mode,
                    mglIssueProgramName(ctx));
        return false;
    }
    mglDrawHostApplyPolygonOffset(renderer, mode);
    if (mode != GL_QUADS &&
        mglSkipIndirectDrawWhenPolygonPointEmulationNeeded(ctx, mode, label)) {
        mglTraceLog("%s skip polygon_point_indirect mode=0x%x program=%u",
                    label ? label : "indirect", (unsigned)mode,
                    mglIssueProgramName(ctx));
        return false;
    }
    return true;
}

static size_t mglIssueIndirectCommandStride(GLsizei stride, size_t defaultStride)
{
    return stride > 0 ? (size_t)stride : defaultStride;
}

static bool mglIssueIndirectCommandOffset(size_t base, size_t index,
                                          size_t stride, size_t *out)
{
    if (!out) {
        return false;
    }
    if (stride != 0u && index > (SIZE_MAX - base) / stride) {
        return false;
    }
    *out = base + index * stride;
    return true;
}

static bool mglIssueArraysIndirectCommandUsable(
    const DrawArraysIndirectCommand *cmd)
{
    if (!cmd || cmd->count == 0u || cmd->instanceCount == 0u) {
        return false;
    }
    return cmd->count <= (uint32_t)INT_MAX &&
           cmd->first <= (uint32_t)INT_MAX &&
           cmd->instanceCount <= (uint32_t)INT_MAX;
}

static bool mglIssueElementsIndirectCommandUsable(
    const DrawElementsIndirectCommand *cmd)
{
    if (!cmd || cmd->count == 0u || cmd->instanceCount == 0u) {
        return false;
    }
    return cmd->count <= (uint32_t)INT_MAX &&
           cmd->instanceCount <= (uint32_t)INT_MAX;
}

static const char *mglIssueArraysIndirectPrepLabel(GLMContext ctx, GLenum mode)
{
    if (mode == GL_PATCHES) {
        return "drawArraysIndirect.patches";
    }
    if (mode == GL_LINE_LOOP) {
        return "drawArraysIndirect.lineLoop";
    }
    if (mode == GL_QUADS) {
        return "drawArraysIndirect.quads";
    }
    if (mglDrawHostHasGeometry(ctx)) {
        return "drawArraysIndirect.geometry";
    }
    return "drawArraysIndirect.cullDistance";
}

static const char *mglIssueElementsIndirectPrepLabel(GLMContext ctx, GLenum mode)
{
    if (mode == GL_PATCHES) {
        return "drawElementsIndirect.patches";
    }
    if (mode == GL_LINE_LOOP) {
        return "drawElementsIndirect.lineLoop";
    }
    if (mode == GL_QUADS) {
        return "drawElementsIndirect.quads";
    }
    if (mglDrawHostHasGeometry(ctx)) {
        return "drawElementsIndirect.geometry";
    }
    return "drawElementsIndirect.cullDistance";
}

static const char *mglIssueMultiArraysIndirectPrepLabel(GLMContext ctx,
                                                        GLenum mode)
{
    if (mode == GL_PATCHES) {
        return "multiDrawArraysIndirect.patches";
    }
    if (mode == GL_LINE_LOOP) {
        return "multiDrawArraysIndirect.lineLoop";
    }
    if (mode == GL_QUADS) {
        return "multiDrawArraysIndirect.quads";
    }
    if (mglDrawHostHasGeometry(ctx)) {
        return "multiDrawArraysIndirect.geometry";
    }
    return "multiDrawArraysIndirect.cullDistance";
}

static const char *mglIssueMultiElementsIndirectPrepLabel(GLMContext ctx,
                                                          GLenum mode)
{
    if (mode == GL_PATCHES) {
        return "multiDrawElementsIndirect.patches";
    }
    if (mode == GL_LINE_LOOP) {
        return "multiDrawElementsIndirect.lineLoop";
    }
    if (mode == GL_QUADS) {
        return "multiDrawElementsIndirect.quads";
    }
    if (mglDrawHostHasGeometry(ctx)) {
        return "multiDrawElementsIndirect.geometry";
    }
    return "multiDrawElementsIndirect.cullDistance";
}

static void mglIssueOneArraysIndirectCommand(GLMContext ctx, void *renderer,
                                             GLenum mode,
                                             const DrawArraysIndirectCommand *cmd,
                                             const char *label)
{
    if (!mglIssueArraysIndirectCommandUsable(cmd)) {
        return;
    }
    mglIssueDrawArrays(ctx, renderer, mode, (GLint)cmd->first,
                       (GLsizei)cmd->count, (GLsizei)cmd->instanceCount,
                       cmd->baseInstance, label);
}

static void mglIssueOneElementsIndirectCommand(
    GLMContext ctx, void *renderer, GLenum mode, GLenum type,
    const DrawElementsIndirectCommand *cmd, const char *label)
{
    if (!mglIssueElementsIndirectCommandUsable(cmd)) {
        return;
    }
    const size_t indexStride = (size_t)mglGLIndexElementSize(type);
    if (indexStride == 0u ||
        (size_t)cmd->first > SIZE_MAX / indexStride) {
        return;
    }
    const void *indices =
        (const void *)(uintptr_t)((size_t)cmd->first * indexStride);
    mglIssueDrawElements(ctx, renderer, mode, (GLsizei)cmd->count, type, indices,
                         (GLsizei)cmd->instanceCount, cmd->baseVertex,
                         cmd->baseInstance, label);
}

extern "C" void mglIssueDrawArraysIndirect(GLMContext ctx, void *renderer,
                                           GLenum mode, const void *indirect,
                                           const char *label)
{
    const char *tag = label ? label : "drawArraysIndirect";
    mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_ENTRY mode=0x%x indirect=%p program=%u",
                (unsigned)mode, indirect, mglIssueProgramName(ctx));
    if (!mglIssueIndirectPreamble(ctx, renderer, mode, tag)) {
        return;
    }
    Buffer *glIndirect = NULL;
    void *metalIndirect = NULL;
    if (!mglDrawHostResolveIndirectBuffer(renderer, ctx, tag, &glIndirect,
                                          &metalIndirect)) {
        mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=resolve_indirect_buffer program=%u",
                    mglIssueProgramName(ctx));
        return;
    }
    const size_t cmdOffset = (size_t)(uintptr_t)indirect;
    if (mglIssueIndirectNeedsCPUExpand(ctx, mode)) {
        if (!mglDrawHostPrepareIndirectCPURead(
                renderer, ctx, mglIssueArraysIndirectPrepLabel(ctx, mode))) {
            return;
        }
        DrawArraysIndirectCommand cmd = {};
        if (!mglReadBufferBytes(glIndirect, metalIndirect, cmdOffset, &cmd,
                                sizeof(cmd),
                                mglIssueArraysIndirectPrepLabel(ctx, mode))) {
            return;
        }
        mglIssueOneArraysIndirectCommand(ctx, renderer, mode, &cmd, tag);
        return;
    }
    if (!mglEncodeDrawArraysIndirectForRenderEncoderOwner(
            mglDrawHostEncoderOwner(renderer), ctx, mode, metalIndirect,
            cmdOffset, tag)) {
        mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=unsupported_mode mode=0x%x program=%u",
                    (unsigned)mode, mglIssueProgramName(ctx));
        return;
    }
    mglDrawHostRecordArraySubmitted(renderer, mode, 0u);
    mglTraceLog("DRAW_ARRAYS_INDIRECT_MTL_SUBMIT path=native mode=0x%x indirect=%p offset=%zu program=%u",
                (unsigned)mode, indirect, cmdOffset, mglIssueProgramName(ctx));
}

extern "C" void mglIssueDrawElementsIndirect(GLMContext ctx, void *renderer,
                                             GLenum mode, GLenum type,
                                             const void *indirect,
                                             const char *label)
{
    const char *tag = label ? label : "drawElementsIndirect";
    mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_ENTRY mode=0x%x type=0x%x indirect=%p program=%u",
                (unsigned)mode, (unsigned)type, indirect,
                mglIssueProgramName(ctx));
    if (!mglIssueIndirectPreamble(ctx, renderer, mode, tag)) {
        return;
    }
    if (mglSkipIndirectElementDrawWhenPrimitiveRestartEnabled(ctx, type, tag)) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=primitive_restart program=%u",
                    mglIssueProgramName(ctx));
        return;
    }
    Buffer *glElement = NULL;
    void *metalElement = NULL;
    if (!mglDrawHostResolveElementBuffer(renderer, ctx, tag, &glElement,
                                         &metalElement)) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=resolve_element_buffer program=%u",
                    mglIssueProgramName(ctx));
        return;
    }
    Buffer *glIndirect = NULL;
    void *metalIndirect = NULL;
    if (!mglDrawHostResolveIndirectBuffer(renderer, ctx, tag, &glIndirect,
                                          &metalIndirect)) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=resolve_indirect_buffer program=%u",
                    mglIssueProgramName(ctx));
        return;
    }
    const size_t cmdOffset = (size_t)(uintptr_t)indirect;
    if (mglIssueIndirectNeedsCPUExpand(ctx, mode)) {
        if (!mglDrawHostPrepareIndirectCPURead(
                renderer, ctx, mglIssueElementsIndirectPrepLabel(ctx, mode))) {
            return;
        }
        DrawElementsIndirectCommand cmd = {};
        if (!mglReadBufferBytes(glIndirect, metalIndirect, cmdOffset, &cmd,
                                sizeof(cmd),
                                mglIssueElementsIndirectPrepLabel(ctx, mode))) {
            return;
        }
        mglIssueOneElementsIndirectCommand(ctx, renderer, mode, type, &cmd,
                                           tag);
        return;
    }
    if (!mglEncodeDrawElementsIndirectForRenderEncoderOwner(
            mglDrawHostEncoderOwner(renderer), ctx, mglDrawHostDevice(renderer),
            glElement, metalElement, mode, type, metalIndirect, cmdOffset,
            tag)) {
        mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=encode program=%u",
                    mglIssueProgramName(ctx));
        return;
    }
    mglDrawHostRecordElementSubmitted(renderer, mode, 0u);
    mglTraceLog("DRAW_ELEMENTS_INDIRECT_MTL_SUBMIT path=native mode=0x%x type=0x%x indirect=%p offset=%zu program=%u",
                (unsigned)mode, (unsigned)type, indirect, cmdOffset,
                mglIssueProgramName(ctx));
}

extern "C" void mglIssueMultiDrawArraysIndirect(GLMContext ctx, void *renderer,
                                                GLenum mode,
                                                const void *indirect,
                                                GLsizei drawcount,
                                                GLsizei stride,
                                                const char *label)
{
    const char *tag = label ? label : "multiDrawArraysIndirect";
    mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_ENTRY mode=0x%x indirect=%p drawcount=%d stride=%d program=%u",
                (unsigned)mode, indirect, (int)drawcount, (int)stride,
                mglIssueProgramName(ctx));
    if (!mglIssueIndirectPreamble(ctx, renderer, mode, tag)) {
        return;
    }
    Buffer *glIndirect = NULL;
    void *metalIndirect = NULL;
    if (!mglDrawHostResolveIndirectBuffer(renderer, ctx, tag, &glIndirect,
                                          &metalIndirect)) {
        mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=resolve_indirect_buffer program=%u",
                    mglIssueProgramName(ctx));
        return;
    }
    if (mglIssueIndirectNeedsCPUExpand(ctx, mode)) {
        if (stride < 0 || drawcount <= 0) {
            return;
        }
        if (!mglDrawHostPrepareIndirectCPURead(
                renderer, ctx,
                mglIssueMultiArraysIndirectPrepLabel(ctx, mode))) {
            return;
        }
        const size_t commandStride = mglIssueIndirectCommandStride(
            stride, sizeof(DrawArraysIndirectCommand));
        const size_t baseOffset = (size_t)(uintptr_t)indirect;
        for (GLsizei i = 0; i < drawcount; ++i) {
            size_t offset = 0u;
            if (!mglIssueIndirectCommandOffset(baseOffset, (size_t)i,
                                               commandStride, &offset)) {
                break;
            }
            DrawArraysIndirectCommand cmd = {};
            if (!mglReadBufferBytes(
                    glIndirect, metalIndirect, offset, &cmd, sizeof(cmd),
                    mglIssueMultiArraysIndirectPrepLabel(ctx, mode))) {
                break;
            }
            mglIssueOneArraysIndirectCommand(ctx, renderer, mode, &cmd, tag);
        }
        return;
    }
    if (drawcount <= 0) {
        return;
    }
    const size_t commandStride = mglIssueIndirectCommandStride(
        stride, sizeof(DrawArraysIndirectCommand));
    const size_t baseOffset = (size_t)(uintptr_t)indirect;
    GLsizei submitted = 0;
    for (GLsizei i = 0; i < drawcount; ++i) {
        size_t offset = 0u;
        if (!mglIssueIndirectCommandOffset(baseOffset, (size_t)i, commandStride,
                                           &offset)) {
            break;
        }
        if (!mglEncodeDrawArraysIndirectForRenderEncoderOwner(
                mglDrawHostEncoderOwner(renderer), ctx, mode, metalIndirect,
                offset, tag)) {
            mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_SKIP reason=unsupported_mode mode=0x%x program=%u",
                        (unsigned)mode, mglIssueProgramName(ctx));
            return;
        }
        submitted++;
    }
    if (submitted > 0) {
        mglDrawHostRecordArraySubmitted(renderer, mode, 0u);
    }
    mglTraceLog("MULTI_DRAW_ARRAYS_INDIRECT_MTL_SUBMIT path=native mode=0x%x indirect=%p drawcount=%d stride=%d program=%u",
                (unsigned)mode, indirect, (int)drawcount, (int)stride,
                mglIssueProgramName(ctx));
}

extern "C" void mglIssueMultiDrawElementsIndirect(
    GLMContext ctx, void *renderer, GLenum mode, GLenum type,
    const void *indirect, GLsizei drawcount, GLsizei stride, const char *label)
{
    const char *tag = label ? label : "multiDrawElementsIndirect";
    mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_ENTRY mode=0x%x type=0x%x indirect=%p drawcount=%d stride=%d program=%u",
                (unsigned)mode, (unsigned)type, indirect, (int)drawcount,
                (int)stride, mglIssueProgramName(ctx));
    if (!mglIssueIndirectPreamble(ctx, renderer, mode, tag)) {
        return;
    }
    if (mglSkipIndirectElementDrawWhenPrimitiveRestartEnabled(ctx, type, tag)) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=primitive_restart program=%u",
                    mglIssueProgramName(ctx));
        return;
    }
    Buffer *glElement = NULL;
    void *metalElement = NULL;
    if (!mglDrawHostResolveElementBuffer(renderer, ctx, tag, &glElement,
                                         &metalElement)) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=resolve_element_buffer program=%u",
                    mglIssueProgramName(ctx));
        return;
    }
    Buffer *glIndirect = NULL;
    void *metalIndirect = NULL;
    if (!mglDrawHostResolveIndirectBuffer(renderer, ctx, tag, &glIndirect,
                                          &metalIndirect)) {
        mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=resolve_indirect_buffer program=%u",
                    mglIssueProgramName(ctx));
        return;
    }
    if (mglIssueIndirectNeedsCPUExpand(ctx, mode)) {
        if (stride < 0 || drawcount <= 0) {
            return;
        }
        if (!mglDrawHostPrepareIndirectCPURead(
                renderer, ctx,
                mglIssueMultiElementsIndirectPrepLabel(ctx, mode))) {
            return;
        }
        const size_t commandStride = mglIssueIndirectCommandStride(
            stride, sizeof(DrawElementsIndirectCommand));
        const size_t baseOffset = (size_t)(uintptr_t)indirect;
        for (GLsizei i = 0; i < drawcount; ++i) {
            size_t offset = 0u;
            if (!mglIssueIndirectCommandOffset(baseOffset, (size_t)i,
                                               commandStride, &offset)) {
                break;
            }
            DrawElementsIndirectCommand cmd = {};
            if (!mglReadBufferBytes(
                    glIndirect, metalIndirect, offset, &cmd, sizeof(cmd),
                    mglIssueMultiElementsIndirectPrepLabel(ctx, mode))) {
                break;
            }
            mglIssueOneElementsIndirectCommand(ctx, renderer, mode, type, &cmd,
                                               tag);
        }
        return;
    }
    if (drawcount <= 0) {
        return;
    }
    const size_t commandStride = mglIssueIndirectCommandStride(
        stride, sizeof(DrawElementsIndirectCommand));
    const size_t baseOffset = (size_t)(uintptr_t)indirect;
    GLsizei submitted = 0;
    for (GLsizei i = 0; i < drawcount; ++i) {
        size_t offset = 0u;
        if (!mglIssueIndirectCommandOffset(baseOffset, (size_t)i, commandStride,
                                           &offset)) {
            break;
        }
        if (!mglEncodeDrawElementsIndirectForRenderEncoderOwner(
                mglDrawHostEncoderOwner(renderer), ctx,
                mglDrawHostDevice(renderer), glElement, metalElement, mode,
                type, metalIndirect, offset, tag)) {
            mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SKIP reason=encode program=%u",
                        mglIssueProgramName(ctx));
            return;
        }
        submitted++;
    }
    if (submitted > 0) {
        mglDrawHostRecordElementSubmitted(renderer, mode, 0u);
    }
    mglTraceLog("MULTI_DRAW_ELEMENTS_INDIRECT_MTL_SUBMIT path=native mode=0x%x type=0x%x indirect=%p drawcount=%d stride=%d program=%u",
                (unsigned)mode, (unsigned)type, indirect, (int)drawcount,
                (int)stride, mglIssueProgramName(ctx));
}


static void mglValidateLogLine(const MGLValidateArraysHostOps *ops, const char *fmt, ...)
{
    if (!ops || !ops->log_line || !fmt) return;
    char buf[768];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    ops->log_line(buf);
}

extern "C" int mglDrawValidateArraysVertexInputs(
    GLMContext ctx, GLenum mode, GLint first, GLsizei count, uint64_t draw_call,
    int validation_enabled, const MGLValidateArraysHostOps *ops)
{
    MGLValidateArraysEarlyStatus st = MGL_VALIDATE_ARRAYS_OK;
    int early_ok = 0;
    uint64_t firstVertex = 0u, lastVertex = 0u;
    const int cont = mglDrawValidateArraysEarly(
        validation_enabled, ctx ? 1 : 0, first, count, &firstVertex, &lastVertex,
        &st, &early_ok);
    if (!cont) {
        if (st == MGL_VALIDATE_ARRAYS_DISABLED) return 1;
        if (st == MGL_VALIDATE_ARRAYS_ZERO_COUNT) return 0;
        if (st == MGL_VALIDATE_ARRAYS_NULL_CTX) {
            mglValidateLogLine(ops, "MGL DRAWARRAYS BLOCK call=%llu reason=null_ctx mode=0x%x first=%d count=%d",
                 (unsigned long long)draw_call, (unsigned)mode, (int)first, (int)count);
            return 0;
        }
        if (st == MGL_VALIDATE_ARRAYS_INVALID_RANGE) {
            mglValidateLogLine(ops, "MGL DRAWARRAYS BLOCK call=%llu reason=invalid_range mode=0x%x first=%d count=%d",
                 (unsigned long long)draw_call, (unsigned)mode, (int)first, (int)count);
            return 0;
        }
        if (st == MGL_VALIDATE_ARRAYS_OVERFLOW) {
            mglValidateLogLine(ops, "MGL DRAWARRAYS BLOCK call=%llu reason=vertex_range_overflow mode=0x%x first=%d count=%d",
                 (unsigned long long)draw_call, (unsigned)mode, (int)first, (int)count);
            return 0;
        }
        return early_ok ? 1 : 0;
    }
    if (!ops || !ops->get_validated_vao || !ops->attrib_enabled ||
        !ops->resolve_attrib || !ops->ensure_mtl_buffer ||
        !ops->mtl_buffer_length || !ops->max_attribs) {
        return 0;
    }

    void *vao = ops->get_validated_vao(ctx, "drawArrays.vboRange");
    if (!vao) {
        mglValidateLogLine(ops, "MGL DRAWARRAYS BLOCK call=%llu reason=invalid_vao mode=0x%x first=%d count=%d",
             (unsigned long long)draw_call, (unsigned)mode, (int)first, (int)count);
        return 0;
    }

    const uint32_t maxAttribs = ops->max_attribs();
    for (uint32_t attrib = 0; attrib < maxAttribs; attrib++) {
        if (!ops->attrib_enabled(vao, attrib)) continue;

        MGLValidateArraysAttribInfo info = {};
        if (!ops->resolve_attrib(ctx, vao, attrib, "drawArrays.vboRange", &info)) {
            mglValidateLogLine(ops, "MGL DRAWARRAYS BLOCK call=%llu attrib=%u reason=invalid_vbo mode=0x%x first=%d count=%d",
                 (unsigned long long)draw_call, (unsigned)attrib, (unsigned)mode,
                 (int)first, (int)count);
            return 0;
        }

        if (!info.has_drawable) {
            mglValidateLogLine(ops, "MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=never_written "
                 "init(source=%u mapped=%u access=0x%x accessFlags=0x%x full=%u "
                 "range=[%lld,%lld) lastOff=%lld lastSize=%lld src=%p hash=0x%016llx)",
                 (unsigned long long)draw_call, (unsigned)attrib,
                 (unsigned)info.buffer_name, (unsigned)info.last_init_source,
                 (unsigned)info.mapped, (unsigned)info.access,
                 (unsigned)info.access_flags, (unsigned)info.has_initialized_data,
                 (long long)info.written_min, (long long)info.written_max,
                 (long long)info.last_write_offset, (long long)info.last_write_size,
                 info.last_write_src_ptr,
                 (unsigned long long)info.last_write_src_hash);
            return 0;
        }

        if (!mglRenderAttribOffsetsValid(info.binding_offset, info.relativeoffset)) {
            mglValidateLogLine(ops, "MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=negative_attrib_offset "
                 "bindingOffset=%lld relativeOffset=%lld",
                 (unsigned long long)draw_call, (unsigned)attrib,
                 (unsigned)info.buffer_name, (long long)info.binding_offset,
                 (long long)info.relativeoffset);
            return 0;
        }

        MGLRenderAttribFetchPlan fetch = {};
        if (!mglRenderPlanAttribFetch(
                info.attrib_type, info.attrib_size, info.stride,
                info.binding_offset, info.relativeoffset, info.divisor,
                firstVertex, lastVertex, info.vbo_size, &fetch) ||
            fetch.status != MGL_ATTRIB_FETCH_OK) {
            const char *reason = "invalid_attrib_format";
            if (fetch.status == MGL_ATTRIB_FETCH_OVERFLOW) reason = "byte_range_overflow";
            else if (fetch.status == MGL_ATTRIB_FETCH_OOB) reason = "vbo_oob";
            mglValidateLogLine(ops, "MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=%s "
                 "byteRange=[%llu,%llu) stride=%llu elem=%llu type=0x%x size=%u divisor=%u",
                 (unsigned long long)draw_call, (unsigned)attrib,
                 (unsigned)info.buffer_name, reason,
                 (unsigned long long)fetch.byte_start, (unsigned long long)fetch.byte_end,
                 (unsigned long long)fetch.stride, (unsigned long long)fetch.elem_bytes,
                 (unsigned)info.attrib_type, (unsigned)info.attrib_size,
                 (unsigned)info.divisor);
            return 0;
        }

        if (!ops->ensure_mtl_buffer(ops->renderer, &info) || !info.mtl_data) {
            mglValidateLogLine(ops, "MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=no_mtl_buffer "
                 "byteRange=[%llu,%llu)",
                 (unsigned long long)draw_call, (unsigned)attrib,
                 (unsigned)info.buffer_name, (unsigned long long)fetch.byte_start,
                 (unsigned long long)fetch.byte_end);
            return 0;
        }

        const uint64_t metalLen = ops->mtl_buffer_length(info.mtl_data);
        if (fetch.byte_end > metalLen) {
            mglValidateLogLine(ops, "MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=metal_oob "
                 "byteRange=[%llu,%llu) metalLen=%llu vboSize=%llu first=%d count=%d",
                 (unsigned long long)draw_call, (unsigned)attrib,
                 (unsigned)info.buffer_name, (unsigned long long)fetch.byte_start,
                 (unsigned long long)fetch.byte_end, (unsigned long long)metalLen,
                 (unsigned long long)info.vbo_size, (int)first, (int)count);
            return 0;
        }

        if (info.written_min >= 0 && info.written_max >= 0) {
            const uint64_t writtenMin = (uint64_t)info.written_min;
            const uint64_t writtenMax = (uint64_t)info.written_max;
            if (fetch.byte_start < writtenMin || fetch.byte_end > writtenMax) {
                mglValidateLogLine(ops, "MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=unwritten_range "
                     "byteRange=[%llu,%llu) written=[%llu,%llu) first=%d count=%d source=%u",
                     (unsigned long long)draw_call, (unsigned)attrib,
                     (unsigned)info.buffer_name, (unsigned long long)fetch.byte_start,
                     (unsigned long long)fetch.byte_end, (unsigned long long)writtenMin,
                     (unsigned long long)writtenMax, (int)first, (int)count,
                     (unsigned)info.last_init_source);
                return 0;
            }
        }

        if (ops->should_inspect && ops->current_program_key && ops->log_line) {
            const uint32_t key = ops->current_program_key(ctx);
            if (ops->should_inspect(draw_call, key) && attrib == 0u) {
                mglValidateLogLine(ops, "MGL TRACE drawArrays.attrib0 call=%llu program=%u buffer=%u first=%d count=%d "
                     "byteRange=[%llu,%llu) vboSize=%llu metalLen=%llu stride=%llu "
                     "bindingOffset=%llu relOffset=%llu elemBytes=%llu",
                     (unsigned long long)draw_call, (unsigned)key,
                     (unsigned)info.buffer_name, (int)first, (int)count,
                     (unsigned long long)fetch.byte_start,
                     (unsigned long long)fetch.byte_end,
                     (unsigned long long)info.vbo_size, (unsigned long long)metalLen,
                     (unsigned long long)fetch.stride,
                     (unsigned long long)info.binding_offset,
                     (unsigned long long)info.relativeoffset,
                     (unsigned long long)fetch.elem_bytes);
            }
        }
    }
    return 1;
}


/* === Indirect-draw skip checks (relocated from mgl_draw_encode.m, O5.4) ===
 * Pure GL-state policy: no Metal types, no ObjC. Callers live in this TU. */

bool mglSkipIndirectElementDrawWhenPrimitiveRestartEnabled(GLMContext ctx,
                                                          GLenum glIndexType,
                                                          const char *label)
{
    uint32_t restartIndex = 0u;
    if (!mglPrimitiveRestartIndexForType(ctx, glIndexType, &restartIndex)) {
        return false;
    }

    static uint64_t s_indirectRestartSkipCount = 0;
    s_indirectRestartSkipCount++;
    if (s_indirectRestartSkipCount <= 8u || (s_indirectRestartSkipCount % 1000u) == 0u) {
        fprintf(stderr, "MGL WARNING: %s primitive restart with indirect indexed draw is not emulated yet type=0x%x restart=%u occurrence=%llu; skipping draw",
              label ? label : "drawElementsIndirect",
              (unsigned)glIndexType,
              (unsigned)restartIndex,
              (unsigned long long)s_indirectRestartSkipCount);
    }
    return true;
}

bool mglSkipIndirectDrawWhenPolygonPointEmulationNeeded(GLMContext ctx,
                                                       GLenum mode,
                                                       const char *label)
{
    if (!mglPolygonModePointForDrawMode(ctx, mode)) {
        return false;
    }

    static uint64_t s_indirectPolygonPointSkipCount = 0;
    s_indirectPolygonPointSkipCount++;
    if (s_indirectPolygonPointSkipCount <= 8u || (s_indirectPolygonPointSkipCount % 1000u) == 0u) {
        fprintf(stderr, "MGL WARNING: %s GL_POLYGON_MODE=GL_POINT requires triangle expansion for indirect draw mode=0x%x occurrence=%llu; skipping draw",
              label ? label : "drawIndirect",
              (unsigned)mode,
              (unsigned long long)s_indirectPolygonPointSkipCount);
    }
    return true;
}
