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
#include "mgl_draw_tess.h"
#include "mgl_frame_activity.h"
#include "mgl_index_buffer.h"
#include "mgl_trace_log.h"
#include "mgl_types_buffer.h"

#include <climits>
#include <cstdint>
#include <cstdio>

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
