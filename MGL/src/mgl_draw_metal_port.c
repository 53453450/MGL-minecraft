/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/* O1.6: true id/MTL materialization ports shared by draw host runners. */

#include <CoreFoundation/CoreFoundation.h>  /* CFRetain / CFRelease */

#include "mgl_encode_context.h"          /* MGLEncodeContext */
#include "mgl_vertex_attrib_query.h"     /* mglRendererGetValidatedVAO */
#include "mgl_vertex_attrib_binding.h"   /* mglRendererResolveVertexAttribBinding */
#include "mgl_draw_validate.h"           /* mglShouldInspectDrawCall */
#include "mgl_env_flag.h"               /* mgl_env_flag_enabled_default_on */
#include "mgl_thread_affinity.h"         /* MGL_ASSERT_GL_THREAD (METAL_LOCK) */
#include "MGLRenderer+DrawSupportUtil.h"
/* Declared next to their definitions in the Objective-C MGLRenderer+Draw_Private.h,
 * which a .c file cannot include; repeated here the way mgl_renderer_ports.c does. */
extern void mglLogDrawWithoutSwapWatchdog(const char *kind, int stage, GLMContext ctx,
                                          void *cb_owner, void *enc_owner,
                                          void *rp_owner);
extern bool mglShouldInspectDrawCall(uint64_t draw_call, GLuint program_name);
/* static inline in the Objective-C header (MGLRenderer+Draw_Private.h): same
 * policy, including the debug-build short circuit. */
static inline bool mglVboRangeValidationEnabled(void)
{
#if defined(DEBUG) || defined(MGL_DEBUG)
    return true;
#else
    return mgl_env_flag_enabled_default_on("MGL_VALIDATE_VBO_RANGE") ? true : false;
#endif
}

/* METAL_LOCK/METAL_UNLOCK were renderer-private macros that only assert the GL
 * thread (see MGLRenderer_Private.h); the C twin keeps the same meaning. */
#define METAL_LOCK()   do { MGL_ASSERT_GL_THREAD(); } while (0)
#define METAL_UNLOCK() do { } while (0)
#include "mgl_draw_tess.h"

#include "mgl_draw_cull.h"
#include "mgl_renderer_ports.h"
#include "mgl_compute_bind.h"    /* compute buffer binding (was a method pair) */
#include "mgl_tess_dispatch.h"   /* TCS dispatch entry (was a shell port) */
#include "mgl_texture_bind.h"     /* mglRendererBindMTLTexture */
#include "mgl_size_constants.h"  /* runtime-array size constants (was a method) */
#include "mgl_draw_support.h"
#include "mgl_ms_sample_loop.h"
#include "mgl_draw_issue.h"
#include "mgl_batch_rt_mark.h"
#include "mgl_index_buffer.h"
#include "mgl_buffer_query.h"
#include "glm_limits.h"
#include "mgl_frame_activity.h"
#include "mgl_draw_gs.h"
#include "mgl_draw_encode.h"
#include "mgl_air_gs_abi.h"
#include "mgl_shader_abi.h"
#include "mgl_renderer_backend.h"
#include <string.h>

/* The Objective-C versions handed these C callers a +1 through
 * (__bridge_retained void *) ...; where the underlying call only lends the
 * object (a borrowed encoder or a cached buffer), that retain has to be
 * explicit now.  The C callers release it with CFRelease
 * (see mglGsMetalEndBlit). */
static void *mglDrawSupportRetainForCaller(void *object)
{
    if (object) {
        CFRetain((CFTypeRef)object);
    }
    return object;
}

void *mglDrawSupportBufferContents(void * buffer)
{
    void *contents = NULL;
    uint64_t length = 0;
    if (!buffer || mglRenderGetBufferContents(
            buffer, &contents, &length) != 0) {
        return NULL;
    }
    return contents;
}

/* AIR stage-out / GS scatter slots carry integers as SIToFP/UIToFP floats.
 * GL transform-feedback stores native int/uint bits — decode before the
 * compact XFB image is published to the GL buffer / CPU shadow.  Packing
 * lives in mglTessPackXFBFieldFromCarrier / mglXfbDecodeIntCarriersInBytes. */

uint64_t mglDrawSupportBufferLength(void * buffer)
{
    MGLRenderBufferInfo info = {0};
    return buffer && mglRenderGetBufferInfo(
        buffer, &info) == 0 ? info.length : 0u;
}

MGLRenderTextureInfo mglDrawSupportTextureInfo(void * texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo(texture, &info);
    }
    return info;
}

bool mglDrawSupportEncodeContextIsActive(
    const MGLEncodeContext *encodeContext)
{
    if (!encodeContext) return false;
    return mglRenderEncoderOwnerHasCurrent(
        encodeContext->render_encoder_owner) == 1;
}


/* mglGeometryGatherIndices → mgl_draw_tess.cpp (O1.4) */

void * mglDrawSupportCreateBuffer(
    void * device,
    size_t length,
    uint64_t options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCreateBuffer(length, options, NULL, &buffer) == 0 &&
        buffer) {
        return buffer;
    }
    return NULL;
}

void * mglDrawSupportCreateBufferWithBytes(
    void * device,
    const void *bytes,
    size_t length,
    uint64_t options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCreateBufferWithBytes(bytes, length, options, NULL,
                                          &buffer) == 0 && buffer) {
        return buffer;
    }
    return NULL;
}

void * mglDrawSupportCreateBlitEncoder(
    void *commandBufferOwner)
{
    return mglRenderCreateBlitEncoderBorrowed(
        commandBufferOwner);
}

void mglDrawSupportBlitCopyBuffer(void * encoder,
                                         void * source,
                                         size_t sourceOffset,
                                         void * destination,
                                         size_t destinationOffset,
                                         size_t size)
{
    (void)mglRenderBlitCopyBuffer(
        encoder, source, sourceOffset,
        destination, destinationOffset, size);
}

void mglDrawSupportEndBlitEncoder(void * encoder)
{
    (void)mglRenderEndBlitEncoder(encoder);
}

void mglDrawSupportSetVertexBuffer(
    void *renderEncoderOwner,
    void * buffer,
    size_t offset,
    size_t index)
{
    (void)mglRenderSetRenderBufferForOwner(
        renderEncoderOwner, buffer, offset,
        MGL_RENDER_BINDING_STAGE_VERTEX, (uint32_t)index);
}

void mglDrawSupportSetVertexBytes(
    void *renderEncoderOwner,
    const void *bytes,
    size_t length,
    size_t index)
{
    (void)mglRenderSetRenderBytesForOwner(
        renderEncoderOwner, bytes, length,
        MGL_RENDER_BINDING_STAGE_VERTEX, (uint32_t)index);
}

void mglDrawSupportDrawIndexedPrimitives(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    size_t indexCount,
    void * indexBuffer,
    size_t indexBufferOffset,
    size_t instanceCount,
    int64_t baseVertex,
    size_t baseInstance)
{
    (void)mglRenderEncodeDrawForRenderEncoderOwner(renderEncoderOwner,
        &(MGLRenderDrawPlan){
            .kind = MGL_RENDER_DRAW_INDEXED,
            .primitive_type = (uint32_t)primitiveType,
            .index_count = indexCount,
            .index_type = (uint32_t)MGL_DRAW_INDEX_UINT32,
            .index_buffer = indexBuffer,
            .index_buffer_offset = indexBufferOffset,
            .instance_count = instanceCount,
            .base_vertex = baseVertex,
            .base_instance = baseInstance,
        }, NULL, 0);
}

void mglDrawSupportDrawPrimitives(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    size_t vertexStart,
    size_t vertexCount,
    size_t instanceCount,
    size_t baseInstance)
{
    MGLRenderDrawPlan plan = {
            .kind = MGL_RENDER_DRAW_ARRAY,
            .primitive_type = (uint32_t)primitiveType,
            .vertex_start = vertexStart,
            .vertex_count = vertexCount,
            .instance_count = instanceCount,
            .base_instance = baseInstance,
        };
    (void)mglRenderEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

void mglDrawSupportDrawPrimitivesIndirect(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    void * indirectBuffer,
    size_t indirectBufferOffset)
{
    MGLRenderDrawPlan plan = {
            .kind = MGL_RENDER_DRAW_ARRAY_INDIRECT,
            .primitive_type = (uint32_t)primitiveType,
            .indirect_buffer = indirectBuffer,
            .indirect_buffer_offset = indirectBufferOffset,
        };
    (void)mglRenderEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

void mglRendererBindCullDistanceEmu(void *renderer, const void *encode_context,
                                    GLenum mode, GLuint first_vertex,
                                    const uint32_t *explicit_vertices,
                                    uint32_t explicit_vertex_count)
{
    if (!renderer || !encode_context) {
        return;
    }
    mglDrawBindCullDistanceEmulationBuffers(renderer, mode, first_vertex,
                                            explicit_vertices,
                                            explicit_vertex_count, encode_context);
}

void * mglDrawSupportCreateComputeEncoder(
    void *commandBufferOwner)
{
    return mglRenderCreateComputeEncoderBorrowed(
        commandBufferOwner);
}

void mglDrawSupportSetComputePipeline(
    void * encoder,
    void * pipeline)
{
    (void)mglRenderSetComputePipelineState(encoder,
                                              pipeline);
}

void mglDrawSupportSetComputeBuffer(
    void * encoder,
    void * buffer,
    size_t offset,
    size_t index)
{
    (void)mglRenderSetComputeBuffer(encoder,
                                       buffer, offset,
                                       (uint32_t)index);
}

void mglDrawSupportSetComputeBytes(
    void * encoder,
    const void *bytes,
    size_t length,
    size_t index)
{
    (void)mglRenderSetComputeBytes(encoder, bytes,
                                      length, (uint32_t)index);
}

void mglDrawSupportDispatchCompute(
    void * encoder,
    uint32_t groupsX,
    uint32_t groupsY,
    uint32_t groupsZ,
    uint32_t threadsX,
    uint32_t threadsY,
    uint32_t threadsZ)
{
    (void)mglRenderDispatchCompute(
        encoder, groupsX, groupsY, groupsZ,
        threadsX, threadsY, threadsZ);
}

void mglDrawSupportEndComputeEncoder(
    void * encoder)
{
    (void)mglRenderEndComputeEncoder(encoder);
}

extern void mglRecordActivePrimitiveQueryDraw(GLMContext ctx,
                                               GLuint64 generated,
                                               GLuint64 written);
extern void mglRecordActivePrimitiveQueryDrawIndexed(GLMContext ctx,
                                                      GLuint index,
                                                      GLuint64 generated,
                                                      GLuint64 written);
extern void mglRecordActiveGeometryShaderQueryDraw(GLMContext ctx,
                                                    GLuint64 invocations,
                                                    GLuint64 primitives);
extern GLboolean mglHasActiveIndexedPrimitiveQuery(GLMContext ctx);
extern GLboolean mglHasActivePrimitiveQuery(GLMContext ctx);
extern GLboolean mglHasActiveGeometryShaderQuery(GLMContext ctx);

void mglRecordGeometryPrimitiveQueries(
    GLMContext ctx,
    GLuint64 generatedStream0,
    GLuint64 writtenStream0,
    bool xfbActive,
    const MGLAIRGSXFBMeta *meta,
    uint32_t streamCount,
    const size_t *bufferWritten,
    const size_t *bufferStride,
    GLuint64 geometryInvocations)
{
    mglRecordActiveGeometryShaderQueryDraw(
        ctx, geometryInvocations, generatedStream0);
    mglRecordActivePrimitiveQueryDraw(
        ctx, generatedStream0,
        mglDrawGsStream0QueryWritten(xfbActive ? 1 : 0, writtenStream0));
    if (!meta || !bufferWritten || !bufferStride) return;
    streamCount = mglDrawGsClampStreamCount(streamCount);
    for (uint32_t s = 1u; s < streamCount; s++) {
        /* Indexed stream s query: generated stays in the meta; written is
         * the ordered scatter's whole-primitive bytes for buffer s divided
         * by its per-record stride (streams > 0 are points, vpp = 1). */
        GLuint64 written = mglDrawGsIndexedStreamWritten(
            xfbActive ? 1 : 0, (uint64_t)bufferWritten[s],
            (uint64_t)bufferStride[s]);
        mglRecordActivePrimitiveQueryDrawIndexed(
            ctx, s, (GLuint64)meta->stream[s].generated, written);
    }
}

void * mglDefaultTessFactorBuffer(void * device,
                                                GLMState *state,
                                                GLuint patchCount)
{
    if (!device || !state || patchCount == 0u) return NULL;
    uint64_t factorBytes = 0u;
    if (!mglTessPlanDefaultFactorBytes(patchCount, &factorBytes)) return NULL;
    void * buffer = mglDrawSupportCreateBuffer(
        device, (size_t)factorBytes, 0u);
    if (!buffer || !mglDrawSupportBufferContents(buffer)) return NULL;

    if (mglRenderFillDefaultTessFactorBuffer(
            (void *)mglDrawSupportBufferContents(buffer),
            factorBytes,
            state->var.patch_default_outer_level,
            state->var.patch_default_inner_level,
            patchCount) != 0) {
        return NULL;
    }
    return buffer;
}

/* Cached variant of the default factor buffer for the TES-only path:
 * consecutive tess draws reuse one stable allocation unless the default
 * patch levels or patch count actually changed. */
void * mglCachedDefaultTessFactorBuffer(
    void * device, MGLRendererBackendHandle *backend, GLMState *state,
    GLuint patchCount)
{
    if (!device || !backend || !state || patchCount == 0u) return NULL;
    float levels[6];
    mglTessFillDefaultFactorLevels(state->var.patch_default_outer_level,
                                   state->var.patch_default_inner_level,
                                   levels);
    void *cached = NULL;
    if (mglRendererBackendGetTessFactorBuffer(
            backend, patchCount, levels, &cached) == 1 && cached) {
        return cached;
    }
    void * fresh = mglDefaultTessFactorBuffer(device, state, patchCount);
    if (!fresh) return NULL;
    if (mglRendererBackendPutTessFactorBuffer(
            backend, patchCount, levels, fresh) != 0) {
        return fresh;
    }
    return fresh;
}

void * mglNativeTessFactorBuffer(void * device,
                                                void * canonical,
                                                GLenum mode,
                                                GLuint patchCount)
{
    if (!device || !canonical || !mglDrawSupportBufferContents(canonical) ||
        patchCount == 0u) {
        return NULL;
    }
    uint32_t repackBytes = 0u;
    const int factorKind = mglTessPlanNativeFactor(
        (uint32_t)mode, (uint64_t)mglDrawSupportBufferLength(canonical),
        patchCount, &repackBytes);
    if (factorKind == MGL_TESS_NATIVE_FACTOR_REUSE) {
        return canonical;
    }
    if (factorKind != MGL_TESS_NATIVE_FACTOR_REPACK_TRI) {
        return NULL;
    }

    void * result = mglDrawSupportCreateBuffer(device, (size_t)repackBytes, 0u);
    if (!result || !mglDrawSupportBufferContents(result)) {
        return NULL;
    }

    if (mglRenderRepackTessFactorTriangles(
            (const void *)mglDrawSupportBufferContents(canonical), (uint64_t)mglDrawSupportBufferLength(canonical),
            (void *)mglDrawSupportBufferContents(result),
            (uint64_t)repackBytes,
            patchCount) != 0) {
        return NULL;
    }
    return result;
}

GLuint64 mglNativeTessPrimitiveCount(void * canonical,
                                             Program *tesProgram,
                                             GLuint patchCount,
                                             GLuint instanceCount)
{
    if (!canonical || !mglDrawSupportBufferContents(canonical) || !tesProgram || patchCount == 0u) {
        return 0u;
    }

    return mglTessGeneratedPrimitiveCount(
        tesProgram, (const void *)mglDrawSupportBufferContents(canonical),
        patchCount, instanceCount);
}


/* O1.2/O1.1: C ports for mglTessRunCaptureSession host ops. */
void mglDrawSupportCaptureMarkDirtyAll(void *ctx_ptr)
{
    GLMContext drawCtx = (GLMContext)ctx_ptr;
    if (drawCtx && drawCtx->active_state) {
        drawCtx->active_state->dirty_bits = DIRTY_ALL;
    }
}

int mglDrawSupportCaptureProcessGL(void *renderer)
{
    return mglRendererProcessGLStatePort(renderer, 1);
}

int mglDrawSupportCaptureEncoderReady(void *renderer)
{
    if (!renderer) return 0;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return mglRenderEncoderOwnerHasCurrent(
               areas.command ? areas.command->currentRenderEncoderOwner : NULL) == 1
               ? 1
               : 0;
}

void mglDrawSupportCaptureBindSlots(void *renderer, void *capture,
                                           const uint32_t *params)
{
    if (!renderer || !capture || !params) return;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    mglTessBindCaptureSlots(
        areas.command ? areas.command->currentRenderEncoderOwner : NULL, capture,
        params);
}

void mglDrawSupportCaptureSetActive(void *renderer, int active)
{
    if (renderer) {
        MGLRendererStateAreas areas;
        mglRendererStateAreasPort(renderer, &areas);
        areas.tessellation->tessVertexCaptureActive = active ? 1 : 0;
    }
}

/* ---- O1.4 HostOps ports (thin MTL / renderer ivar materialization) ---- */


/* C-ABI accessors for the Metal-facing sub-objects (see declarations in
 * MGLRenderer_Private.h). Defined here so they compile with full knowledge of
 * the MGLRenderer class extension; consumed by file-scope C functions in this
 * and the other encode ports. */

/* C port (ObjC-zeroing T4): the render pass state owner of a renderer handle,
 * so C modules do not need a category to read it. */


static void mglStageMarkCbHasWork(void *renderer)
{
    if (!renderer) return;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    if (areas.batching) areas.batching->currentCommandBufferHasWork = 1;
}

static void mglStageFlushCB(void *renderer, int wait)
{
    if (!renderer) return;
    mglRendererFlushCommandBufferPort(renderer, wait ? 1 : 0);
}

static void *mglStageBufContents(void *buffer)
{
    return mglDrawSupportBufferContents(buffer);
}

static void *mglDrawPortVertexCaptureArray(void *renderer, GLMContext ctx,
                                             GLint first, GLsizei count,
                                             GLsizei instanceCount,
                                             GLuint baseInstance,
                                             uint64_t *out_offset); /* below */

static void *mglStageCaptureArray(void *renderer, GLMContext ctx, GLint first,
                                  GLsizei count, GLsizei instanceCount,
                                  GLuint baseInstance, uint64_t *out_offset)
{
    return mglDrawPortVertexCaptureArray(renderer, ctx, first, count,
                                         instanceCount, baseInstance,
                                         out_offset);
}

static void *mglDrawPortVertexCaptureIndexed(
    void *renderer, GLMContext ctx, void *index_mtl, GLenum indexType,
    uint64_t index_offset, GLsizei count, GLint baseVertex,
    GLsizei instanceCount, GLuint baseInstance, uint32_t maxIndex,
    uint64_t *out_offset); /* below */

static void *mglStageCaptureIndexed(void *renderer, GLMContext ctx, void *index_mtl,
                                    GLenum indexType, uint64_t index_offset,
                                    GLsizei count, GLint baseVertex,
                                    GLsizei instanceCount, GLuint baseInstance,
                                    uint32_t maxIndex, uint64_t *out_offset)
{
    return mglDrawPortVertexCaptureIndexed(renderer, ctx, index_mtl, indexType,
                                           index_offset, count, baseVertex,
                                           instanceCount, baseInstance, maxIndex,
                                           out_offset);
}

static int mglStageBindProgram(void *renderer, Program *program)
{
    return mglRendererBindMTLProgramPort(renderer, program);
}

static int mglStageProcessBuffer(void *renderer, Buffer *buf)
{
    return mglRendererProcessBuffer(renderer, buf) ? 1 : 0;
}

static void *mglStageCreateBuffer(void *renderer, uint64_t length)
{
    if (!renderer) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    void *buf = mglDrawSupportCreateBuffer(mglRendererBackendGetDevice(areas.backend), (size_t)length, 0u);
    return buf;
}

static void *mglStageCreateBufferBytes(void *renderer, const void *bytes,
                                       uint64_t length)
{
    if (!renderer) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    void *buf = mglDrawSupportCreateBufferWithBytes(mglRendererBackendGetDevice(areas.backend), bytes,
                                                 (size_t)length, 0u);
    return buf;
}

static void *mglStageCachedFactors(void *renderer, GLMContext ctx,
                                   uint32_t patch_count)
{
    if (!renderer || !ctx) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    void *buf = mglCachedDefaultTessFactorBuffer(mglRendererBackendGetDevice(areas.backend), areas.backend,
                                              ctx->active_state, patch_count);
    /* Cached on backend — borrow only. */
    return buf;
}

static void *mglStageNativeFactors(void *renderer, void *canonical, GLenum mode,
                                   uint32_t patch_count)
{
    if (!renderer) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    void *buf = mglNativeTessFactorBuffer(mglRendererBackendGetDevice(areas.backend), canonical,
                                       mode, patch_count);
    return mglDrawSupportRetainForCaller(buf);
}

static int mglStageDispatchTCS(void *renderer, GLMContext ctx, Program *tcs,
                               MGLAIRTessDrawContract *contract)
{
    /* The TCS entry is C now (log 126); the shell port that forwarded to the
     * Objective-C method is retired. */
    return mglTessDispatchControlShader(renderer, ctx, tcs, contract) ? 1 : 0;
}

static int mglStageDispatchAirTES(void *renderer, GLMContext ctx, Program *tes,
                                  MGLAIRTessDrawContract *contract,
                                  uint32_t patch_count, GLsizei instanceCount,
                                  GLuint baseInstance)
{
    return mglRendererDispatchAIRTessEvalComputePort(
        renderer, ctx, tes, contract, patch_count, (int32_t)instanceCount,
        baseInstance);
}

static int mglStageDispatchAirTESVertex(void *renderer, GLMContext ctx,
                                        Program *tes,
                                        MGLAIRTessDrawContract *contract,
                                        uint32_t patch_count, GLsizei instanceCount,
                                        GLuint baseInstance)
{
    /* The TES-vertex render entry is C now (log 127); the shell port that
     * forwarded to the Objective-C method is retired. */
    return mglTessDispatchAIRTessEvalVertexRender(renderer, ctx, tes, contract,
                                                  patch_count, instanceCount,
                                                  baseInstance)
               ? 1
               : 0;
}

static int mglStageProcessGL(void *renderer)
{
    return mglRendererProcessGLStatePort(renderer, 1);
}

static int mglStageEncoderHasCurrent(void *renderer)
{
    if (!renderer) return 0;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return mglRenderEncoderOwnerHasCurrent(
               areas.command ? areas.command->currentRenderEncoderOwner : NULL)
               ? 1
               : 0;
}

static int mglStageRasterEmpty(void *renderer)
{
    return renderer ? mglDrawRasterizationIsEmpty(renderer) : 0;
}

static int mglStageFullyCulled(void *renderer, GLenum mode)
{
    return renderer ? mglDrawModeIsFullyCulled(renderer, (uint32_t)mode) : 0;
}

static void mglStageApplyPolygonOffset(void *renderer, GLenum mode)
{
    if (renderer) {
        mglDrawApplyPolygonOffset(renderer, (uint32_t)mode);
    }
}

static void mglStageEndRender(void *renderer)
{
    if (renderer) mglRendererEndRenderEncodingPort(renderer);
}

static void mglStageClearNativeCB(void *renderer)
{
    MGLRendererStateAreas areas;
    if (!renderer) return;
    mglRendererStateAreasPort(renderer, &areas);
    mglRendererClearStageBindingCopyBacksPort(renderer, &areas.tessellation->nativeTESCopyBacks);
}

static int mglStageFlushNativeCB(void *renderer)
{
    MGLRendererStateAreas areas;
    if (!renderer) return 0;
    mglRendererStateAreasPort(renderer, &areas);
    return mglRendererFlushStageBindingCopyBacksPort(
        renderer, &areas.tessellation->nativeTESCopyBacks, 0);
}

static void mglStageBeginNativeTES(void *renderer, Program *tes)
{
    if (!renderer) return;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    areas.tessellation->nativeTESProgram = tes;
    areas.tessellation->nativeTESActive = 1;
}

static void mglStageEndNativeTES(void *renderer)
{
    if (!renderer) return;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    areas.tessellation->nativeTESActive = 0;
    areas.tessellation->nativeTESProgram = NULL;
}

static void mglStageResetTessDrawState(void *renderer)
{
    if (!renderer) return;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    (void)mglRendererBackendSetTessVertexCaptureBuffer(areas.backend, NULL);
    areas.tessellation->tessVertexCaptureOffset = 0u;
    (void)mglRendererBackendSetTessControlPointIndexBuffer(areas.backend, NULL);
    areas.tessellation->tessIndexedDraw = 0;
    areas.tessellation->tessInstanceRecords = 0u;
    (void)mglRendererBackendSetTcsOutputBuffer(areas.backend, NULL);
    areas.tessellation->tcsOutputOffset = 0u;
    areas.tessellation->tcsOutputStride = 0u;
    areas.tessellation->tcsOutVertices = 0u;
    (void)mglRendererBackendSetCurrentTessFactorBuffer(areas.backend, NULL);
}

static void mglStageSetTessCapture(void *renderer, void *buf, uint64_t offset,
                                   uint64_t instance_records, int indexed)
{
    if (!renderer) return;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    (void)mglRendererBackendSetTessVertexCaptureBuffer(areas.backend, buf);
    areas.tessellation->tessVertexCaptureOffset = (size_t)offset;
    areas.tessellation->tessIndexedDraw = indexed ? 1 : 0;
    areas.tessellation->tessInstanceRecords = (size_t)instance_records;
    if (!buf) {
        areas.tessellation->tessIndexedDraw = 0;
        areas.tessellation->tessInstanceRecords = 0u;
        areas.tessellation->tessVertexCaptureOffset = 0u;
        (void)mglRendererBackendSetTessControlPointIndexBuffer(areas.backend,
                                                               NULL);
    }
}

static void mglStageSetControlPointIndex(void *renderer, void *gather)
{
    if (!renderer) return;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    (void)mglRendererBackendSetTessControlPointIndexBuffer(areas.backend,
                                                           gather);
}

static void mglStageAdoptCaptureAsTCS(void *renderer, uint32_t stride,
                                      uint32_t out_vertices)
{
    if (!renderer) return;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    void *cap = mglRendererBackendGetTessVertexCaptureBuffer(areas.backend);
    (void)mglRendererBackendSetTcsOutputBuffer(areas.backend, cap);
    areas.tessellation->tcsOutputOffset =
        areas.tessellation->tessVertexCaptureOffset;
    areas.tessellation->tcsOutputStride = stride;
    areas.tessellation->tcsOutVertices = out_vertices;
}

static void mglStageSetCurrentFactors(void *renderer, void *factors)
{
    if (!renderer) return;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    (void)mglRendererBackendSetCurrentTessFactorBuffer(areas.backend, factors);
}

static void *mglStageGetTessCapture(void *renderer)
{
    if (!renderer) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return mglRendererBackendGetTessVertexCaptureBuffer(areas.backend);
}

static void *mglStageGetTcsOutput(void *renderer)
{
    if (!renderer) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return mglRendererBackendGetTcsOutputBuffer(areas.backend);
}

static void *mglStageGetFactors(void *renderer)
{
    if (!renderer) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return mglRendererBackendGetCurrentTessFactorBuffer(areas.backend);
}

static void *mglStageGetPatchOut(void *renderer)
{
    if (!renderer) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return mglRendererBackendGetTcsPatchOutBuffer(areas.backend);
}

static void *mglStageGetControlPointIndex(void *renderer)
{
    if (!renderer) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return mglRendererBackendGetTessControlPointIndexBuffer(areas.backend);
}

static void *mglStageEncoderOwner(void *renderer)
{
    if (!renderer) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return areas.command ? areas.command->currentRenderEncoderOwner : NULL;
}

static uint32_t mglStageGetTcsOutVerts(void *renderer)
{
    if (!renderer) return 0u;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return (uint32_t)areas.tessellation->tcsOutVertices;
}

static uint64_t mglStageGetTcsOutStride(void *renderer)
{
    if (!renderer) return 0u;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return (uint64_t)areas.tessellation->tcsOutputStride;
}

static uint64_t mglStageGetTessCaptureOff(void *renderer)
{
    if (!renderer) return 0u;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return (uint64_t)areas.tessellation->tessVertexCaptureOffset;
}

static uint64_t mglStageGetTessInstRecords(void *renderer)
{
    if (!renderer) return 0u;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return (uint64_t)areas.tessellation->tessInstanceRecords;
}

static int mglStageGetTessIndexed(void *renderer)
{
    if (!renderer) return 0;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return areas.tessellation->tessIndexedDraw ? 1 : 0;
}

static uint64_t mglStageNativePrimCount(void *canonical, Program *tes,
                                        uint32_t patch_count,
                                        uint32_t instance_count)
{
    return mglNativeTessPrimitiveCount(canonical, tes, patch_count,
                                       instance_count);
}

static void mglStageRecordQuery(GLMContext ctx, uint64_t generated,
                                uint64_t written)
{
    mglRecordActivePrimitiveQueryDraw(ctx, generated, written);
}

static void mglStageLogError(const char *msg)
{
    if (msg) fprintf(stderr, "%s\n", msg);
}

static int mglStageEnsurePassthrough(void *renderer, Program *program,
                                     uint32_t output_primitive)
{
    return mglRendererEnsureAIRGeometryPassthroughPort(renderer, program,
                                                       output_primitive);
}

static int mglStagePendingGsActive(void *renderer)
{
    if (!renderer) return 0;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return areas.tessellation->pendingGSInputActive ? 1 : 0;
}

static void *mglStagePendingGsInput(void *renderer)
{
    if (!renderer) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return areas.tessellation->pendingGSInput;
}

static uint32_t mglStagePendingGsOff(void *renderer)
{
    if (!renderer) return 0u;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return (uint32_t)areas.tessellation->pendingGSInputOffset;
}

static uint32_t mglStagePendingGsStride(void *renderer)
{
    if (!renderer) return 0u;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return (uint32_t)areas.tessellation->pendingGSInputStride;
}


/* ---- A1 / O1.4: GS Metal expansion HostOps (thin MTL materialization only) ---- */

static void mglGsMetalRelease(void *obj)
{
    if (obj) CFRelease(obj);
}

static uint64_t mglGsMetalBufferLength(void *buffer)
{
    return buffer ? (uint64_t)mglDrawSupportBufferLength(buffer) : 0u;
}

static int mglGsMetalEnsureCB(void *renderer)
{
    return renderer ? mglPlatformShellNewCommandBuffer(renderer) : 0;
}

static int mglGsMetalBindDrawTextures(void *renderer, GLMContext ctx)
{
    if (!renderer || !ctx) return 0;
    for (size_t unit = 0; unit < TEXTURE_UNITS; unit++) {
        Texture *image = ctx->active_state->image_units[unit].tex;
        Texture *sampled = ctx->active_state->active_textures[unit];
        if (image && !mglRendererBindMTLTexture(renderer, image)) return 0;
        if (sampled && !mglRendererBindMTLTexture(renderer, sampled)) return 0;
    }
    return 1;
}

static void *mglGsMetalMtlForBuffer(void *renderer, Buffer *buf)
{
    if (!renderer || !buf) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    if (!buf->data.mtl_data) {
        mglRendererBindMTLBuffer(renderer, buf);
    }
    return buf->data.mtl_data;
}

static int mglGsMetalFillComputeBindings(void *renderer, GLMContext ctx,
                                         MGLRenderComputeExecutionPlan *plan,
                                         MGLRenderCopyBackEntry *copybacks,
                                         uint32_t copybacks_cap,
                                         uint32_t *copybacks_count,
                                         void **temporaries_out)
{
    if (!renderer || !plan || !copybacks || !copybacks_count) return 0;
    (void)ctx;
    if (temporaries_out) *temporaries_out = NULL;
    MGLStageBindingCopyBackList stageCopyBacks = {0};
    void *temps = mglRendererTemporariesCreate();
    void *compute = NULL;
    bool buffersOK = mglComputeBindBuffersToEncoder(
        renderer, _GEOMETRY_SHADER, compute, &stageCopyBacks, plan, temps);
    bool texturesOK = buffersOK && mglComputeBindTexturesToEncoder(
        renderer, _GEOMETRY_SHADER, compute, plan, temps);
    if (!buffersOK || !texturesOK) {
        if (compute) mglDrawSupportEndComputeEncoder(compute);
        mglRendererClearStageBindingCopyBacksPort(renderer, &stageCopyBacks);
        mglRendererTemporariesRelease(temps);
        return 0;
    }
    uint32_t n = mglRenderCollectCopyBackEntries(
        (const MGLRenderCopyBackEntry *)stageCopyBacks.slots,
        kMGLMaxBufferSlots, copybacks, copybacks_cap);
    *copybacks_count = n;
    mglRendererClearStageBindingCopyBacksPort(renderer, &stageCopyBacks);
    /* The plan only stores borrowed MTL pointers, and this function returns
     * before the C++ side encodes/dispatches it.  Hand the keep-alive set back
     * as a +1 reference so the caller can hold it across the encode; releasing
     * it here would drop every temporary it retains, and setBuffer: would then
     * retain a deallocated buffer (EXC_BAD_ACCESS / "message sent to
     * deallocated instance").  (The Objective-C version used an
     * NSMutableArray here; the C entry hands back the same +1 shape.) */
    if (temporaries_out) {
        *temporaries_out = temps;
    } else {
        mglRendererTemporariesRelease(temps);
    }
    return 1;
}

static void *mglGsMetalCmdOwner(void *renderer)
{
    if (!renderer) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return areas.command ? areas.command->currentCommandBufferOwner : NULL;
}

static void *mglGsMetalRecoveryOwner(void *renderer)
{
    if (!renderer) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return areas.gpu_recovery_command_owner ? *areas.gpu_recovery_command_owner
                                            : NULL;
}

static void mglGsMetalNoteDeviceReset(void *renderer)
{
    if (!renderer) return;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    if (areas.core) {
        atomic_store_explicit(&areas.core->deviceResetRequested, true,
                              memory_order_release);
    }
}

static void mglGsMetalSetExpansion(void *renderer, Program *program, int active,
                                   GLenum last_draw_mode)
{
    if (!renderer) return;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    areas.geometry->expansionActive = active ? 1 : 0;
    areas.geometry->program = active ? program : NULL;
    if (active && last_draw_mode) {
        areas.core->lastDrawPrimitiveMode = last_draw_mode;
    }
}

static void *mglGsMetalBeginBlit(void *renderer)
{
    if (!renderer) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    void *blit = mglDrawSupportCreateBlitEncoder(
        areas.command ? areas.command->currentCommandBufferOwner : NULL);
    return mglDrawSupportRetainForCaller(blit);
}

static void mglGsMetalBlitCopy(void *blit, void *src, uint64_t src_off, void *dst,
                               uint64_t dst_off, uint64_t bytes)
{
    mglDrawSupportBlitCopyBuffer(blit, src,
                                 (size_t)src_off, dst,
                                 (size_t)dst_off, (size_t)bytes);
}

static void mglGsMetalEndBlit(void *blit)
{
    if (!blit) return;
    mglDrawSupportEndBlitEncoder(blit);
    CFRelease(blit);
}

static void *mglGsMetalBindingOwner(void *renderer)
{
    if (!renderer) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return areas.binding_state_owner ? *areas.binding_state_owner : NULL;
}

/* A1: clears are in mgl_draw_gs_metal.cpp; ObjC only rebinds MTL resources. */
static int mglGsMetalRebindFragment(void *renderer, GLMContext ctx)
{
    if (!renderer || !ctx) return 0;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLEncodeContext gsEncCtx = {
        .render_encoder_owner =
            areas.command ? areas.command->currentRenderEncoderOwner : NULL,
    };
    (void)mglRendererBindFragmentBuffersToCurrentRenderEncoderPort(renderer,
                                                                   &gsEncCtx);
    (void)mglRendererBindBufferSizeConstantsForRenderEncoder(renderer);
    Program *gsVertexProgram =
        mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    Program *gsFragmentProgram =
        mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    return mglRendererBindStorageImagesForVertexProgramPort(
        renderer, gsVertexProgram, gsFragmentProgram);
}

static void mglGsMetalRecordQueries(GLMContext ctx, uint64_t generated,
                                    uint64_t written, int xfb_active,
                                    const MGLAIRGSXFBMeta *meta,
                                    uint32_t stream_count,
                                    const uint64_t *buffer_written,
                                    const uint64_t *buffer_stride,
                                    uint64_t geometry_invocations)
{
    size_t bw[MGL_AIR_GS_MAX_STREAMS] = {0};
    size_t bs[MGL_AIR_GS_MAX_STREAMS] = {0};
    uint32_t n = stream_count < MGL_AIR_GS_MAX_STREAMS ? stream_count
                                                       : MGL_AIR_GS_MAX_STREAMS;
    for (uint32_t i = 0; i < n; i++) {
        if (buffer_written) bw[i] = (size_t)buffer_written[i];
        if (buffer_stride) bs[i] = (size_t)buffer_stride[i];
    }
    mglRecordGeometryPrimitiveQueries(ctx, generated, written,
                                      xfb_active ? 1 : 0, meta, stream_count,
                                      bw, bs, geometry_invocations);
}

/* The capture session itself lives in the shell TU
 * (mglPlatformShellGpuCapture{Start,Stop}); this file only publishes them as
 * the host-ops callbacks. */
static void mglGsMetalGpuCaptureStart(void *renderer)
{
    mglPlatformShellGpuCaptureStart(renderer);
}

static void mglGsMetalGpuCaptureStop(void *renderer)
{
    mglPlatformShellGpuCaptureStop(renderer);
}

static void mglGsMetalSetVertexBuffer(void *encoder_owner, void *buffer,
                                      uint64_t offset, uint32_t index)
{
    mglDrawSupportSetVertexBuffer(encoder_owner, buffer,
                                  (size_t)offset, index);
}

static void mglGsMetalDrawPrims(void *encoder_owner, uint32_t output_primitive,
                                uint32_t vertex_start, uint32_t vertex_count,
                                uint32_t instance_count, uint32_t base_instance)
{
    mglDrawSupportDrawPrimitives(encoder_owner, output_primitive, vertex_start,
                                 vertex_count, instance_count, base_instance);
}

static void mglGsMetalDrawPrimsIndirect(void *encoder_owner,
                                        uint32_t output_primitive, void *counts,
                                        uint64_t offset)
{
    mglDrawSupportDrawPrimitivesIndirect(encoder_owner, output_primitive,
                                         counts,
                                         (size_t)offset);
}

static void mglGsMetalLogDiag(const char *msg)
{
    if (msg) fprintf(stderr, "%s\n", msg);
}

/* A1: fill nested Metal expansion HostOps (no ObjC expansion middle-man). */
static MGLGsMetalExpansionHostOps mglGsMetalMakeExpansionOps(void *renderer)
{
    return (MGLGsMetalExpansionHostOps){
        .renderer = renderer,
        .create_buffer = mglStageCreateBuffer,
        .create_buffer_with_bytes = mglStageCreateBufferBytes,
        .buffer_contents = mglStageBufContents,
        .buffer_length = mglGsMetalBufferLength,
        .release = mglGsMetalRelease,
        .ensure_command_buffer = mglGsMetalEnsureCB,
        .bind_draw_textures = mglGsMetalBindDrawTextures,
        .mtl_for_buffer = mglGsMetalMtlForBuffer,
        .fill_compute_bindings = mglGsMetalFillComputeBindings,
        .command_buffer_owner = mglGsMetalCmdOwner,
        .recovery_owner = mglGsMetalRecoveryOwner,
        .note_device_reset = mglGsMetalNoteDeviceReset,
        .set_expansion = mglGsMetalSetExpansion,
        .mark_cb_has_work = mglStageMarkCbHasWork,
        .begin_blit = mglGsMetalBeginBlit,
        .blit_copy = mglGsMetalBlitCopy,
        .end_blit = mglGsMetalEndBlit,
        .process_gl_state = mglStageProcessGL,
        .encoder_has_current = mglStageEncoderHasCurrent,
        .raster_empty = mglStageRasterEmpty,
        .fully_culled = mglStageFullyCulled,
        .apply_polygon_offset = mglStageApplyPolygonOffset,
        .binding_state_owner = mglGsMetalBindingOwner,
        .rebind_fragment_after_gs = mglGsMetalRebindFragment,
        .encoder_owner = mglStageEncoderOwner,
        .flush_command_buffer = mglStageFlushCB,
        .record_queries = mglGsMetalRecordQueries,
        .gpu_capture_start = mglGsMetalGpuCaptureStart,
        .gpu_capture_stop = mglGsMetalGpuCaptureStop,
        .set_vertex_buffer = mglGsMetalSetVertexBuffer,
        .draw_primitives = mglGsMetalDrawPrims,
        .draw_primitives_indirect = mglGsMetalDrawPrimsIndirect,
        .log_diag = mglGsMetalLogDiag,
    };
}





/* ---- Port runners: fill HostOps and call C++ domain orchestration ---- */

static int mglStagePrimitiveRestart(GLMContext ctx, GLenum index_type,
                                    uint32_t *out_restart_index)
{
    return mglPrimitiveRestartIndexForType(ctx, index_type, out_restart_index)
               ? 1
               : 0;
}

static void *mglStagePrepareElementIndex(void *renderer, void *index_buffer,
                                         GLenum gl_index_type,
                                         uint64_t *inout_offset,
                                         uint64_t *inout_mtl_type)
{
    if (!renderer || !index_buffer || !inout_offset || !inout_mtl_type) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    size_t off = (size_t)*inout_offset;
    uint64_t typ = *inout_mtl_type;
    void *prepared = mglPreparedElementIndexBuffer(
        mglRendererBackendGetDevice(areas.backend), NULL,
        index_buffer, gl_index_type, &off,
        &typ);
    if (!prepared) return NULL;
    *inout_offset = (uint64_t)off;
    *inout_mtl_type = typ;
    return prepared;
}

static void mglStageSetCtx(void *renderer, GLMContext ctx)
{
    if (renderer) mglPlatformShellSetContext(renderer, ctx);
}

static void mglStageClearCullCapture(void *renderer)
{
    if (!renderer) return;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    (void)mglRendererBackendSetCullDistanceCaptureBuffer(areas.backend, NULL);
    areas.tessellation->cullDistanceCaptureFirstInstance = 0u;
    areas.tessellation->cullDistanceCaptureInstanceStride = 0u;
}

static void mglStageSetCullCaptureActive(void *renderer, int active)
{
    if (!renderer) return;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    areas.tessellation->cullDistanceCaptureActive = active ? 1 : 0;
}

static void mglStageStoreCullCapture(void *renderer, void *buf,
                                     uint32_t first_instance,
                                     uint32_t instance_stride)
{
    if (!renderer) return;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    (void)mglRendererBackendSetCullDistanceCaptureBuffer(areas.backend, buf);
    areas.tessellation->cullDistanceCaptureFirstInstance = first_instance;
    areas.tessellation->cullDistanceCaptureInstanceStride = instance_stride;
}

static void *mglStageLoadCullCapture(void *renderer)
{
    if (!renderer) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return mglRendererBackendGetCullDistanceCaptureBuffer(areas.backend);
}

static void *mglStageDevicePtr(void *renderer)
{
    if (!renderer) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return mglRendererBackendGetDevice(areas.backend);
}

static void mglStageBindCullEmu(void *renderer, GLenum mode, GLuint first_vertex,
                                const uint32_t *explicit_vertices,
                                uint32_t explicit_vertex_count,
                                const void *enc_ctx)
{
    if (!renderer || !enc_ctx) return;
    mglDrawBindCullDistanceEmulationBuffers(renderer, mode, first_vertex,
                                            explicit_vertices,
                                            explicit_vertex_count, enc_ctx);
}

static int mglStageTryArraySplitEncode(void *renderer, void *device,
                                       void *encoder_owner, GLenum mode,
                                       GLint first, GLsizei count,
                                       uint64_t instance_count,
                                       uint64_t base_instance,
                                       const void *enc_ctx)
{
    return mglEncodeCullDistanceArraySplitForRenderEncoderOwner(
               encoder_owner, device, mode, first,
               count, (size_t)instance_count,
               (size_t)base_instance, renderer, enc_ctx,
               mglRendererBindCullDistanceEmu)
               ? 1
               : 0;
}

static void mglStageDrawIndexedPrimsPort(void *encoder_owner,
                                         uint32_t primitive_type,
                                         uint64_t index_count,
                                         void *index_buffer,
                                         uint64_t index_offset,
                                         uint64_t instance_count,
                                         int32_t base_vertex,
                                         uint64_t base_instance)
{
    mglDrawSupportDrawIndexedPrimitives(
        encoder_owner, primitive_type, (size_t)index_count,
        index_buffer, (size_t)index_offset,
        (size_t)instance_count, (int64_t)base_vertex,
        (size_t)base_instance);
}

static int mglStageEncodeCtxActive(const void *enc_ctx)
{
    return mglDrawSupportEncodeContextIsActive((const MGLEncodeContext *)enc_ctx)
               ? 1
               : 0;
}

static MGLTessVertexCaptureHostOps mglStageMakeVertexCaptureOps(void *renderer)
{
    return (MGLTessVertexCaptureHostOps){
        .renderer = renderer,
        .bind_mtl_program = mglStageBindProgram,
        .create_buffer = mglStageCreateBuffer,
        .create_buffer_with_bytes = mglStageCreateBufferBytes,
        .buffer_contents = mglStageBufContents,
        .buffer_length = mglGsMetalBufferLength,
        .mark_dirty_all = mglDrawSupportCaptureMarkDirtyAll,
        .process_gl_state = mglStageProcessGL,
        .encoder_has_current = mglStageEncoderHasCurrent,
        .bind_capture_slots = mglDrawSupportCaptureBindSlots,
        .set_capture_active = mglDrawSupportCaptureSetActive,
        .encoder_owner = mglStageEncoderOwner,
        .mark_cb_has_work = mglStageMarkCbHasWork,
        .end_render_encoding = mglStageEndRender,
        .primitive_restart = mglStagePrimitiveRestart,
        .prepare_element_index = mglStagePrepareElementIndex,
        .log_gs_diag = mglStageLogError,
    };
}

static MGLCullDistanceHostOps mglStageMakeCullOps(void *renderer)
{
    return (MGLCullDistanceHostOps){
        .renderer = renderer,
        .bind_mtl_program = mglStageBindProgram,
        .create_buffer = mglStageCreateBuffer,
        .set_ctx = mglStageSetCtx,
        .clear_cull_capture = mglStageClearCullCapture,
        .set_cull_capture_active = mglStageSetCullCaptureActive,
        .store_cull_capture = mglStageStoreCullCapture,
        .load_cull_capture = mglStageLoadCullCapture,
        .mark_dirty_all = mglDrawSupportCaptureMarkDirtyAll,
        .process_gl_state = mglStageProcessGL,
        .encoder_has_current = mglStageEncoderHasCurrent,
        .encoder_owner = mglStageEncoderOwner,
        .device = mglStageDevicePtr,
        .mark_cb_has_work = mglStageMarkCbHasWork,
        .end_render_encoding = mglStageEndRender,
        .bind_cull_emu = mglStageBindCullEmu,
        .try_array_split_encode = mglStageTryArraySplitEncode,
        .draw_indexed_primitives = mglStageDrawIndexedPrimsPort,
        .encode_context_active = mglStageEncodeCtxActive,
        .primitive_restart = mglStagePrimitiveRestart,
    };
}

void *mglDrawHostRunVertexCaptureArray(void *renderer, GLMContext ctx,
                                       GLint first, GLsizei count,
                                       GLsizei instanceCount,
                                       GLuint baseInstance,
                                       uint64_t *out_offset)
{
    if (!renderer) return NULL;
    MGLTessVertexCaptureHostOps ops = mglStageMakeVertexCaptureOps(renderer);
    return mglTessRunVertexCaptureArray(ctx, first, count, instanceCount,
                                        baseInstance, out_offset, &ops);
}
static void *mglDrawPortVertexCaptureArray(void *renderer, GLMContext ctx,
                                           GLint first, GLsizei count,
                                           GLsizei instanceCount,
                                           GLuint baseInstance,
                                           uint64_t *out_offset)
{
    return mglDrawHostRunVertexCaptureArray(renderer, ctx, first, count,
                                            instanceCount, baseInstance,
                                            out_offset);
}

void *mglDrawHostRunVertexCaptureIndexed(
    void *renderer, GLMContext ctx, void *index_mtl, uint64_t indexType,
    uint64_t index_offset, GLsizei count, GLint baseVertex,
    GLsizei instanceCount, GLuint baseInstance, uint32_t maxIndex,
    uint64_t *out_offset)
{
    if (!renderer) return NULL;
    MGLTessVertexCaptureHostOps ops = mglStageMakeVertexCaptureOps(renderer);
    return mglTessRunVertexCaptureIndexed(ctx, index_mtl, indexType, index_offset,
                                          count, baseVertex, instanceCount,
                                          baseInstance, maxIndex, out_offset,
                                          &ops);
}
static void *mglDrawPortVertexCaptureIndexed(
    void *renderer, GLMContext ctx, void *index_mtl, GLenum indexType,
    uint64_t index_offset, GLsizei count, GLint baseVertex,
    GLsizei instanceCount, GLuint baseInstance, uint32_t maxIndex,
    uint64_t *out_offset)
{
    return mglDrawHostRunVertexCaptureIndexed(
        renderer, ctx, index_mtl, indexType, index_offset, count, baseVertex,
        instanceCount, baseInstance, maxIndex, out_offset);
}

static void *mglStageGetValidatedVAOPort(GLMContext ctx, const char *where)
{
    return (void *)mglRendererGetValidatedVAO(ctx, where);
}

static int mglStageAttribEnabledPort(void *vao, uint32_t attrib)
{
    VertexArray *v = (VertexArray *)vao;
    if (!v || attrib >= (uint32_t)MAX_ATTRIBS) return 0;
    return (v->enabled_attribs & (0x1u << attrib)) != 0u ? 1 : 0;
}

static int mglStageResolveAttribPort(GLMContext ctx, void *vao, uint32_t attrib,
                                     const char *where,
                                     MGLValidateArraysAttribInfo *out)
{
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    MGLResolvedVertexAttribBinding resolved = {0};
    if (!mglRendererResolveVertexAttribBinding(ctx, (VertexArray *)vao, attrib,
                                               where, &resolved)) {
        return 0;
    }
    const VertexAttrib *a = resolved.attrib;
    Buffer *vbo = resolved.buffer;
    if (!a || !vbo) return 0;
    out->attrib_index = attrib;
    out->buffer_name = vbo->name;
    out->binding_offset = resolved.binding_offset;
    out->relativeoffset = resolved.relativeoffset;
    out->stride = resolved.stride;
    out->divisor = resolved.divisor;
    out->attrib_type = (uint32_t)a->type;
    out->attrib_size = (uint32_t)a->size;
    out->vbo_size = vbo->size;
    out->has_drawable = mglRendererBufferHasDrawableContents(vbo) ? 1 : 0;
    out->written_min = vbo->written_min;
    out->written_max = vbo->written_max;
    out->last_init_source = vbo->last_init_source;
    out->mapped = vbo->mapped;
    out->access = vbo->access;
    out->access_flags = vbo->access_flags;
    out->has_initialized_data = vbo->has_initialized_data;
    out->last_write_offset = vbo->last_write_offset;
    out->last_write_size = vbo->last_write_size;
    out->last_write_src_ptr = vbo->last_write_src_ptr;
    out->last_write_src_hash = vbo->last_write_src_hash;
    out->buffer_obj = vbo;
    out->mtl_data = vbo->data.mtl_data;
    return 1;
}

static int mglStageEnsureMtlBufferPort(void *renderer,
                                       MGLValidateArraysAttribInfo *info)
{
    if (!renderer || !info || !info->buffer_obj) return 0;
    Buffer *vbo = (Buffer *)info->buffer_obj;
    if (!vbo->data.mtl_data) {
        mglRendererBindMTLBuffer(renderer, vbo);
    }
    info->mtl_data = vbo->data.mtl_data;
    return info->mtl_data ? 1 : 0;
}

static uint64_t mglStageMtlLenPort(void *mtl_data)
{
    return mtl_data ? mglDrawSupportBufferLength(mtl_data) : 0u;
}

static uint32_t mglStageMaxAttribsPort(void) { return (uint32_t)MAX_ATTRIBS; }

static uint32_t mglStageProgramKeyPort(GLMContext ctx)
{
    return (uint32_t)mglCurrentRenderProgramKey(ctx);
}

static int mglStageShouldInspectPort(uint64_t draw_call, uint32_t program_key)
{
    return mglShouldInspectDrawCall(draw_call, (GLuint)program_key) ? 1 : 0;
}

static void mglStageValidateLogPort(const char *msg)
{
    if (msg) fprintf(stderr, "%s\n", msg);
}

static MGLValidateArraysHostOps mglStageMakeValidateOps(void *renderer)
{
    return (MGLValidateArraysHostOps){
        .renderer = renderer,
        .get_validated_vao = mglStageGetValidatedVAOPort,
        .attrib_enabled = mglStageAttribEnabledPort,
        .resolve_attrib = mglStageResolveAttribPort,
        .ensure_mtl_buffer = mglStageEnsureMtlBufferPort,
        .mtl_buffer_length = mglStageMtlLenPort,
        .max_attribs = mglStageMaxAttribsPort,
        .current_program_key = mglStageProgramKeyPort,
        .should_inspect = mglStageShouldInspectPort,
        .log_line = mglStageValidateLogPort,
    };
}

/* O1.4: host ABI entry points fill HostOps and call C++ runners. */

bool mglDrawHostHandleXFB(void *renderer, GLMContext ctx, GLenum mode,
                          GLint first, GLsizei count, GLsizei instanceCount,
                          GLuint baseInstance)
{
    if (!renderer) return false;
    MGLXfbVsDrawHostOps ops = {
        .renderer = renderer,
        .capture_vs_positions = mglStageCaptureArray,
        .mark_cb_has_work = mglStageMarkCbHasWork,
        .flush_command_buffer = mglStageFlushCB,
        .buffer_contents = mglStageBufContents,
        .dispatch_error = NULL,
    };
    return mglXfbRunVsOnlyDraw(ctx, mode, first, count, instanceCount,
                               baseInstance, &ops) != 0;
}

bool mglDrawHostHandleTessellation(void *renderer, GLMContext ctx,
                                   GLenum *mode, GLint first, GLsizei count,
                                   GLenum indexType, const void *indices,
                                   GLint baseVertex, GLsizei instanceCount,
                                   GLuint baseInstance, const char *label)
{
    if (!renderer || !mode) return false;
    mglPlatformShellSetContext(renderer, ctx);
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLTessPatchDrawHostOps ops = {
        .renderer = renderer,
        .device = (mglRendererBackendGetDevice(areas.backend)),
        .bind_mtl_program = mglStageBindProgram,
        .capture_array = mglStageCaptureArray,
        .capture_indexed = mglStageCaptureIndexed,
        .process_buffer = mglStageProcessBuffer,
        .flush_command_buffer = mglStageFlushCB,
        .mark_cb_has_work = mglStageMarkCbHasWork,
        .create_buffer = mglStageCreateBuffer,
        .create_buffer_with_bytes = mglStageCreateBufferBytes,
        .buffer_contents = mglStageBufContents,
        .cached_default_factors = mglStageCachedFactors,
        .native_factor_buffer = mglStageNativeFactors,
        .dispatch_tcs = mglStageDispatchTCS,
        .dispatch_air_tes = mglStageDispatchAirTES,
        .dispatch_air_tes_vertex = mglStageDispatchAirTESVertex,
        .process_gl_state = mglStageProcessGL,
        .encoder_has_current = mglStageEncoderHasCurrent,
        .raster_empty = mglStageRasterEmpty,
        .fully_culled = mglStageFullyCulled,
        .apply_polygon_offset = mglStageApplyPolygonOffset,
        .end_render_encoding = mglStageEndRender,
        .clear_native_copybacks = mglStageClearNativeCB,
        .flush_native_copybacks = mglStageFlushNativeCB,
        .begin_native_tes = mglStageBeginNativeTES,
        .end_native_tes = mglStageEndNativeTES,
        .reset_tess_draw_state = mglStageResetTessDrawState,
        .set_tess_vertex_capture = mglStageSetTessCapture,
        .set_control_point_index_buffer = mglStageSetControlPointIndex,
        .adopt_capture_as_tcs_output = mglStageAdoptCaptureAsTCS,
        .set_current_factors = mglStageSetCurrentFactors,
        .get_tess_vertex_capture = mglStageGetTessCapture,
        .get_tcs_output = mglStageGetTcsOutput,
        .get_current_factors = mglStageGetFactors,
        .get_tcs_patch_out = mglStageGetPatchOut,
        .get_control_point_index = mglStageGetControlPointIndex,
        .encoder_owner = mglStageEncoderOwner,
        .get_tcs_out_vertices = mglStageGetTcsOutVerts,
        .get_tcs_output_stride = mglStageGetTcsOutStride,
        .get_tess_capture_offset = mglStageGetTessCaptureOff,
        .get_tess_instance_records = mglStageGetTessInstRecords,
        .get_tess_indexed_draw = mglStageGetTessIndexed,
        .native_primitive_count = mglStageNativePrimCount,
        .record_primitive_query = mglStageRecordQuery,
        .dispatch_error = NULL,
        .log_error = mglStageLogError,
    };
    return mglTessRunPatchDraw(ctx, mode, first, count, indexType, indices,
                               baseVertex, instanceCount, baseInstance, label,
                               &ops) != 0;
}

bool mglDrawHostHandleGeometry(void *renderer, GLMContext ctx, GLenum mode,
                               GLint first, GLsizei count, GLenum indexType,
                               const void *indices, GLint baseVertex,
                               GLsizei instanceCount, GLuint baseInstance,
                               const char *label)
{
    if (!renderer) return false;
    mglPlatformShellSetContext(renderer, ctx);
    MGLGsMetalExpansionHostOps metal_ops = mglGsMetalMakeExpansionOps(renderer);
    MGLGsDrawHostOps ops = {
        .renderer = renderer,
        .bind_mtl_program = mglStageBindProgram,
        .ensure_passthrough = mglStageEnsurePassthrough,
        .process_buffer = mglStageProcessBuffer,
        .capture_array = mglStageCaptureArray,
        .capture_indexed = mglStageCaptureIndexed,
        .create_buffer_with_bytes = mglStageCreateBufferBytes,
        .pending_gs_input_active = mglStagePendingGsActive,
        .pending_gs_input = mglStagePendingGsInput,
        .pending_gs_input_offset = mglStagePendingGsOff,
        .pending_gs_input_stride = mglStagePendingGsStride,
        .metal_ops = &metal_ops,
        .dispatch_error = NULL,
        .log_diag = NULL,
    };
    return mglDrawGsRunDraw(ctx, mode, first, count, indexType, indices,
                            baseVertex, instanceCount, baseInstance, label,
                            &ops) != 0;
}





/* Blocks in the MS sample loop became function pointers plus a context. */
typedef struct {
    GLMContext ctx; void *renderer; uint32_t mode; int32_t first; int32_t count;
    int32_t instance_count; uint32_t base_instance; const char *label;
} MGLMsDrawArraysOnce;

static void mglMsDrawArraysOnce(void *v)
{
    MGLMsDrawArraysOnce *o = v;
    mglIssueDrawArrays(o->ctx, o->renderer, o->mode, o->first, o->count,
                       o->instance_count, o->base_instance, o->label);
}

typedef struct {
    GLMContext ctx; void *renderer; uint32_t mode; int32_t count; uint32_t type;
    const void *indices; int32_t instance_count; int32_t base_vertex;
    uint32_t base_instance; const char *label;
} MGLMsDrawElementsOnce;

static void mglMsDrawElementsOnce(void *v)
{
    MGLMsDrawElementsOnce *o = v;
    mglIssueDrawElements(o->ctx, o->renderer, o->mode, o->count, o->type,
                         o->indices, o->instance_count, o->base_vertex,
                         o->base_instance, o->label);
}

void mglDrawHostGuardIssueArrays(void *renderer, GLMContext ctx, GLenum mode,
                                 GLint first, GLsizei count,
                                 GLsizei instanceCount, GLuint baseInstance,
                                 const char *label, int with_ms)
{
    if (!renderer || !ctx) {
        return;
    }
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    METAL_LOCK();
    areas.core->lastDrawPrimitiveMode = mode;
    if (with_ms) {
        MGLMsDrawArraysOnce once = {.ctx = ctx, .renderer = renderer, .mode = mode,
                                    .first = first, .count = count,
                                    .instance_count = instanceCount,
                                    .base_instance = baseInstance,
                                    .label = label};
        if (mglRendererRunEmulatedMSSampleDrawLoopIfNeeded(
                renderer, ctx, mglMsDrawArraysOnce, &once)) {
            METAL_UNLOCK();
            return;
        }
    }
    mglIssueDrawArrays(ctx, renderer, mode, first, count, instanceCount,
                       baseInstance, label);
    if (with_ms) {
        mglRendererBroadcastEmulatedMSSamplePlanesAfterDrawIfNeeded(renderer, ctx);
    }
    METAL_UNLOCK();
}

void mglDrawHostGuardIssueElements(void *renderer, GLMContext ctx, GLenum mode,
                                   GLsizei count, GLenum type,
                                   const void *indices, GLsizei instanceCount,
                                   GLint baseVertex, GLuint baseInstance,
                                   const char *label, int with_ms)
{
    if (!renderer || !ctx) {
        return;
    }
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    METAL_LOCK();
    areas.core->lastDrawPrimitiveMode = mode;
    if (with_ms) {
        MGLMsDrawElementsOnce once = {.ctx = ctx, .renderer = renderer, .mode = mode,
                                      .count = count, .type = type,
                                      .indices = indices,
                                      .instance_count = instanceCount,
                                      .base_vertex = baseVertex,
                                      .base_instance = baseInstance,
                                      .label = label};
        if (mglRendererRunEmulatedMSSampleDrawLoopIfNeeded(
                renderer, ctx, mglMsDrawElementsOnce, &once)) {
            METAL_UNLOCK();
            return;
        }
    }
    mglIssueDrawElements(ctx, renderer, mode, count, type, indices,
                         instanceCount, baseVertex, baseInstance, label);
    if (with_ms) {
        mglRendererBroadcastEmulatedMSSamplePlanesAfterDrawIfNeeded(renderer, ctx);
    }
    METAL_UNLOCK();
}

bool mglDrawHostBindContext(void *renderer, GLMContext ctx)
{
    if (!renderer) {
        return false;
    }
    mglPlatformShellSetContext(renderer, ctx);
    return true;
}

void mglDrawHostSetLastPrimitiveMode(void *renderer, GLenum mode)
{
    if (!renderer) return;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    {
        areas.core->lastDrawPrimitiveMode = mode;
    }
}

/* mglDrawHostHandle* tess → StageHost O1.4 C++ runners */

/* mglDrawHostHandle* gs → StageHost O1.4 C++ runners */

/* mglDrawHostHandle* xfb → StageHost O1.4 C++ runners */

bool mglDrawHostCaptureCullDistanceArray(void *renderer, GLMContext ctx,
                                         GLint first, GLsizei count,
                                         GLsizei instanceCount,
                                         GLuint baseInstance)
{
    if (!renderer) return false;
    MGLCullDistanceHostOps ops = mglStageMakeCullOps(renderer);
    return mglDrawRunCullDistanceArrayCapture(ctx, first, count, instanceCount,
                                              baseInstance, &ops) != 0;
}

bool mglDrawHostCaptureCullDistanceElement(void *renderer, GLMContext ctx,
                                           const uint8_t *indexBytes,
                                           GLenum indexType, GLsizei count,
                                           GLint baseVertex,
                                           GLsizei instanceCount,
                                           GLuint baseInstance)
{
    if (!renderer) return false;
    MGLCullDistanceHostOps ops = mglStageMakeCullOps(renderer);
    return mglDrawRunCullDistanceElementCapture(ctx, indexBytes, indexType, count,
                                                baseVertex, instanceCount,
                                                baseInstance, &ops) != 0;
}

bool mglDrawHostProcessGLStateLocked(void *renderer, bool draw_command)
{
    if (!renderer) {
        return false;
    }
    return mglRendererProcessGLStateLockedPort(renderer, draw_command) ? true : false;
}

bool mglDrawHostRasterizationIsEmpty(void *renderer)
{
    return renderer ? mglDrawRasterizationIsEmpty(renderer) : 0;
}

bool mglDrawHostModeFullyCulled(void *renderer, GLenum mode)
{
    return renderer
               ? mglDrawModeIsFullyCulled(renderer, (uint32_t)mode) : 0;
}

void mglDrawHostApplyPolygonOffset(void *renderer, GLenum mode)
{
    if (renderer) {
        mglDrawApplyPolygonOffset(renderer, (uint32_t)mode);
    }
}

bool mglDrawHostEnsureRasterEncoder(void *renderer)
{
    return mglRendererEnsureRasterEncoderForDrawPort(renderer) ? true : false;
}

bool mglDrawHostValidateArrayVertexInputs(void *renderer, GLMContext ctx,
                                          GLenum mode, GLint first,
                                          GLsizei count)
{
    if (!renderer) return false;
    MGLValidateArraysHostOps ops = mglStageMakeValidateOps(renderer);
    const int enabled = mglVboRangeValidationEnabled() ? 1 : 0;
    return mglDrawValidateArraysVertexInputs(ctx, mode, first, count, 0ull,
                                             enabled, &ops) != 0;
}

bool mglDrawHostEncodeCullDistanceArray(void *renderer, GLenum mode,
                                        GLint first, GLsizei count,
                                        GLsizei instanceCount,
                                        GLuint baseInstance)
{
    if (!renderer) return false;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    if (mglPolygonModePointForDrawMode(areas.ctx, mode)) {
        return false;
    }
    MGLEncodeContext encCtx = {
        .render_encoder_owner =
            areas.command ? areas.command->currentRenderEncoderOwner : NULL,
    };
    MGLCullDistanceHostOps ops = mglStageMakeCullOps(renderer);
    return mglDrawEncodeCullDistanceArray(areas.ctx, mode, first, count,
                                          instanceCount, baseInstance, &encCtx,
                                          &ops) != 0;
}

void *mglDrawHostEncoderOwner(void *renderer)
{
    if (!renderer) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return areas.command ? areas.command->currentRenderEncoderOwner : NULL;
}

void *mglDrawHostDevice(void *renderer)
{
    if (!renderer) return NULL;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return mglRendererBackendGetDevice(areas.backend);
}

void mglDrawHostRecordArraySubmitted(void *renderer, GLenum mode,
                                     uint64_t vertexCount)
{
    if (!renderer) return;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    {
        mglBatchRecordArrayDrawSubmitted(renderer, areas.ctx, mode, vertexCount);
    }
}

void mglDrawHostWatchdogArrays(void *renderer, GLMContext ctx)
{
    if (!renderer) {
        return;
    }
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    mglLogDrawWithoutSwapWatchdog(
        "arrays", 0, ctx,
        areas.command ? areas.command->currentCommandBufferOwner : NULL,
        areas.command ? areas.command->currentRenderEncoderOwner : NULL,
        areas.command ? areas.command->renderPassStateOwner : NULL);
}


bool mglDrawHostEncodeCullDistanceElementBytes(
    void *renderer, GLenum mode, const uint8_t *indexBytes, GLenum type,
    GLsizei count, GLint baseVertex, GLsizei instanceCount, GLuint baseInstance,
    int polygon_line_mode, const void *enc_ctx)
{
    if (!renderer) return false;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCullDistanceHostOps ops = mglStageMakeCullOps(renderer);
    GLMContext ctx = areas.ctx;
    return mglDrawEncodeCullDistanceElement(
               ctx, mode, indexBytes, type, count, baseVertex, instanceCount,
               baseInstance, polygon_line_mode, enc_ctx, &ops) != 0;
}

bool mglDrawHostPrepareEncodeCullDistanceElement(
    void *renderer, GLenum mode, const uint8_t *indexBytes, GLenum type,
    GLsizei count, GLint baseVertex, GLsizei instanceCount, GLuint baseInstance,
    int polygon_line_mode)
{
    if (!renderer) return false;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    if (!areas.ctx) return false;
    MGLCullDistanceHostOps ops = mglStageMakeCullOps(renderer);
    return mglDrawPrepareAndEncodeCullDistanceElement(
               areas.ctx, mode, indexBytes, type, count, baseVertex,
               instanceCount, baseInstance, polygon_line_mode, &ops) != 0;
}

bool mglDrawHostEncodeCullDistanceElements(void *renderer, GLenum mode,
                                           GLenum type, const void *indices,
                                           GLsizei count, GLint baseVertex,
                                           GLsizei instanceCount,
                                           GLuint baseInstance)
{
    if (!renderer) return false;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    if (mglPolygonModePointForDrawMode(areas.ctx, mode)) {
        return false;
    }
    Buffer *glBuffer = NULL;
    void *metalBuffer = NULL;
    if (!mglRendererResolveElementBufferForDraw(renderer,
                                                "drawElements", areas.ctx,
                                                &glBuffer, &metalBuffer)) {
        return false;
    }
    const size_t offset = (size_t)(uintptr_t)indices;
    const uint8_t *cullIndexBytes = mglElementIndexSourceForDraw(
        glBuffer, metalBuffer, type, offset,
        count);
    MGLCullDistanceHostOps ops = mglStageMakeCullOps(renderer);
    return mglDrawPrepareAndEncodeCullDistanceElement(
               areas.ctx, mode, cullIndexBytes, type, count, baseVertex,
               instanceCount, baseInstance,
               mglPolygonModeLineForDrawMode(areas.ctx, mode) ? 1 : 0, &ops) != 0;
}

bool mglDrawHostResolveElementBuffer(void *renderer, GLMContext ctx,
                                     const char *label, Buffer **glBufferOut,
                                     void **metalBufferOut)
{
    if (!renderer) {
        return false;
    }
    void *metalBuffer = NULL;
    if (!mglRendererResolveElementBufferForDraw(renderer,
                                                label ? label : "drawElements",
                                                ctx, glBufferOut,
                                                &metalBuffer)) {
        return false;
    }
    if (metalBufferOut) {
        *metalBufferOut = metalBuffer;
    }
    return true;
}

void mglDrawHostRecordElementSubmitted(void *renderer, GLenum mode,
                                       uint64_t indexCount)
{
    if (!renderer) return;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    {
        mglBatchRecordElementDrawSubmitted(renderer, areas.ctx, mode, indexCount);
    }
}

void mglDrawHostWatchdogElements(void *renderer, GLMContext ctx)
{
    if (!renderer) {
        return;
    }
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    mglLogDrawWithoutSwapWatchdog(
        "elements", 0, ctx,
        areas.command ? areas.command->currentCommandBufferOwner : NULL,
        areas.command ? areas.command->currentRenderEncoderOwner : NULL,
        areas.command ? areas.command->renderPassStateOwner : NULL);
}

bool mglDrawHostResolveIndirectBuffer(void *renderer, GLMContext ctx,
                                      const char *label, Buffer **glBufferOut,
                                      void **metalBufferOut)
{
    if (!renderer) {
        return false;
    }
    void *metalBuffer = NULL;
    if (!mglDrawResolveIndirectBuffer(renderer, label ? label : "indirectDraw",
                                      ctx, glBufferOut, &metalBuffer)) {
        return false;
    }
    if (metalBufferOut) {
        *metalBufferOut = metalBuffer;
    }
    return true;
}

bool mglDrawHostPrepareIndirectCPURead(void *renderer, GLMContext ctx,
                                       const char *label)
{
    return mglRendererPrepareEmulatedIndirectCPUReadPort(
               renderer, ctx, label ? label : "indirectDraw") ? true : false;
}

bool mglDrawHostHasGeometry(GLMContext ctx)
{
    Program *gsProgram = mglResolveProgramForStageFromState(ctx, _GEOMETRY_SHADER);
    return gsProgram && gsProgram->shader_slots[_GEOMETRY_SHADER];
}

bool mglDrawHostUsesCullDistance(GLMContext ctx)
{
    Program *vertexProgram =
        mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    return vertexProgram && vertexProgram->uses_cull_distance;
}


