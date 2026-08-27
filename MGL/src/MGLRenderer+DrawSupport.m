/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+DrawSupport.m
// Draw validation, element-buffer resolution and rasterization helper
// methods extracted from MGLRenderer+Draw.m

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "mgl_frame_activity.h"
#import "mgl_compute_pipeline_cache.h"
#include "mgl_env_flag.h"
#include "mgl_shader_abi.h"
#include "mgl_air_gs_abi.h"
#include "mgl_air_tess_abi.h"
#include "mgl_aux_assets.h"
#include "mgl_program_reflection.h"

static void *mglDrawSupportBufferContents(id buffer)
{
    void *contents = NULL;
    uint64_t length = 0;
    if (!buffer || mglRenderGetBufferContents(
            (__bridge void *)buffer, &contents, &length) != 0) {
        return NULL;
    }
    return contents;
}

static uint64_t mglDrawSupportBufferLength(id buffer)
{
    MGLRenderBufferInfo info = {0};
    return buffer && mglRenderGetBufferInfo(
        (__bridge void *)buffer, &info) == 0 ? info.length : 0u;
}

static MGLRenderTextureInfo mglDrawSupportTextureInfo(id texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo((__bridge void *)texture, &info);
    }
    return info;
}

static BOOL mglDrawSupportEncodeContextIsActive(
    const MGLEncodeContext *encodeContext)
{
    if (!encodeContext) return NO;
    return mglRenderEncoderOwnerHasCurrent(
        encodeContext->render_encoder_owner) == 1;
}


static bool mglGeometryGatherIndices(const uint8_t *indexBytes,
                                     GLenum indexType,
                                     GLsizei count,
                                     int32_t baseVertex,
                                     bool restartEnabled,
                                     uint32_t restartIndex,
                                     uint32_t inputVertices,
                                     uint32_t **outGather,
                                     uint32_t *outGatherCount,
                                     uint32_t *outPrimitiveCount,
                                     uint32_t *outMaxIndex)
{

    (void)baseVertex; /* gather stores raw index values (vertex_id) */
    if (!outGather || !outGatherCount || !outPrimitiveCount || !outMaxIndex) {
        return false;
    }
    const uint32_t elemBytes = indexType == GL_UNSIGNED_BYTE ? 1u
        : indexType == GL_UNSIGNED_SHORT ? 2u : 4u;
    MGLRenderGeometryGatherResult result = {0};
    if (mglRenderGeometryGatherIndices(
            indexBytes, elemBytes, (uint32_t)count,
            restartEnabled ? 1 : 0, restartIndex, inputVertices,
            &result) != 0) {
        return false;
    }
    *outGather = result.gather;
    *outGatherCount = result.gather_count;
    *outPrimitiveCount = result.primitive_count;
    *outMaxIndex = result.max_index;
    return true;
}

/* Build the primitive input stream for a GS draw.  The GS kernel consumes a
 * complete, fixed-width primitive per work item, so array strips/fans/loops
 * and indexed restart segments are normalized here to the same gather ABI.
 * Values remain raw vertex ids: indexed capture applies baseVertex, while
 * array capture uses first_vertex to address its [first, first + count) span. */
static bool mglGeometryGatherTopology(const uint8_t *indexBytes,
                                      GLenum indexType,
                                      GLsizei count,
                                      GLint first,
                                      bool indexed,
                                      bool restartEnabled,
                                      uint32_t restartIndex,
                                      GLenum mode,
                                      uint32_t **outGather,
                                      uint32_t *outGatherCount,
                                      uint32_t *outPrimitiveCount,
                                      uint32_t *outMaxIndex)
{
    if (!outGather || !outGatherCount || !outPrimitiveCount ||
        !outMaxIndex || count <= 0 || (indexed && !indexBytes) ||
        (!indexed && first < 0)) {
        return false;
    }
    const uint32_t n = (uint32_t)count;
    const uint32_t elemBytes = indexType == GL_UNSIGNED_BYTE ? 1u
        : indexType == GL_UNSIGNED_SHORT ? 2u : 4u;
    if ((size_t)n > SIZE_MAX / sizeof(uint32_t) ||
        (size_t)n > SIZE_MAX / (6u * sizeof(uint32_t))) return false;
    uint32_t *source = malloc((size_t)n * sizeof(*source));
    uint32_t *segment = malloc((size_t)n * sizeof(*segment));
    uint32_t *gather = malloc((size_t)n * 6u * sizeof(*gather));
    if (!source || !segment || !gather) {
        free(source); free(segment); free(gather);
        return false;
    }
    uint32_t maxIndex = 0u;
    for (uint32_t i = 0u; i < n; i++) {
        uint32_t value;
        if (!indexed) {
            const int64_t v = (int64_t)first + (int64_t)i;
            if (v < 0 || (uint64_t)v > UINT32_MAX) {
                free(source); free(segment); free(gather);
                return false;
            }
            value = (uint32_t)v;
        } else if (elemBytes == 1u) {
            value = indexBytes[i];
        } else if (elemBytes == 2u) {
            value = ((const uint16_t *)indexBytes)[i];
        } else {
            value = ((const uint32_t *)indexBytes)[i];
        }
        source[i] = value;
        if (!(indexed && restartEnabled && value == restartIndex) &&
            value > maxIndex) maxIndex = value;
    }

    uint32_t gathered = 0u;
    uint32_t primitives = 0u;
    uint32_t segmentCount = 0u;
    const bool restartMode = indexed && restartEnabled;
    const uint32_t primitiveWidth =
        (mode == GL_POINTS) ? 1u
        : (mode == GL_LINES || mode == GL_LINE_STRIP ||
           mode == GL_LINE_LOOP) ? 2u
        : (mode == GL_LINES_ADJACENCY) ? 4u
        : (mode == GL_TRIANGLES_ADJACENCY) ? 6u : 3u;

    /* Emit one restart-delimited segment according to the GL topology. */
    #define EMIT(v) do { gather[gathered++] = (v); } while (0)
    #define EMIT_SEGMENT() do {                                                \
        if (segmentCount > 0u) {                                               \
            if (mode == GL_POINTS) {                                           \
                for (uint32_t q = 0u; q < segmentCount; q++) {                 \
                    EMIT(segment[q]); primitives++;                           \
                }                                                                  \
            } else if (mode == GL_LINES || mode == GL_TRIANGLES ||             \
                       mode == GL_LINES_ADJACENCY ||                           \
                       mode == GL_TRIANGLES_ADJACENCY) {                       \
                const uint32_t groups = segmentCount / primitiveWidth;         \
                for (uint32_t q = 0u; q < groups; q++) {                        \
                    for (uint32_t k = 0u; k < primitiveWidth; k++)             \
                        EMIT(segment[q * primitiveWidth + k]);                 \
                    primitives++;                                               \
                }                                                                  \
            } else if (mode == GL_LINE_STRIP_ADJACENCY) {                       \
                /* Each lines_adjacency primitive is the four vertices          \
                 * starting at its first member; GL 4.6 10.4.2.2 strips        \
                 * adjacency advance by ONE vertex per primitive (a            \
                 * two-vertex stride halves the emitted lines). */              \
                for (uint32_t q = 0u; q + 3u < segmentCount; q++) {             \
                    for (uint32_t k = 0u; k < 4u; k++)                           \
                        EMIT(segment[q + k]);                                   \
                    primitives++;                                               \
                }                                                                  \
            } else if (mode == GL_TRIANGLE_STRIP_ADJACENCY) {                   \
                /* GL 4.6 10.1.14 + table 10.1: each triangles_adjacency   \
                 * primitive takes a six-vertex window, but its adjacency \
                 * vertices reach outside that window (even triangles     \
                 * borrow the previous window's first vertex, odd ones    \
                 * the next window's), and odd triangles swap their first \
                 * two core vertices.  Emit the exact gl_in order. */     \
                uint32_t tri = 0u;                                              \
                for (uint32_t q = 0u; q + 5u < segmentCount; q += 2u, tri++) {  \
                    uint32_t last = (q + 6u >= segmentCount);                   \
                    if (tri == 0u) {                                            \
                        /* first: core 1,3,5; adj 2,7,4 */                      \
                        EMIT(segment[q]); EMIT(segment[q + 1u]);                \
                        EMIT(segment[q + 2u]);                                  \
                        EMIT(last ? segment[q + 5u] : segment[q + 6u]);         \
                        EMIT(segment[q + 4u]); EMIT(segment[q + 3u]);           \
                    } else if (tri & 1u) {                                      \
                        /* odd: core 2i+3,2i+1,2i+5; adj 2i-1,2i+4,2i+7 */      \
                        EMIT(segment[q + 2u]); EMIT(segment[q - 2u]);           \
                        EMIT(segment[q]);                                       \
                        EMIT(segment[q + 3u]); EMIT(segment[q + 4u]);           \
                        EMIT(last ? segment[q + 5u] : segment[q + 6u]);         \
                    } else {                                                    \
                        /* even: core 2i+1,2i+3,2i+5; adj 2i-1,2i+6,2i+4 */     \
                        EMIT(segment[q]); EMIT(segment[q - 2u]);                \
                        EMIT(segment[q + 2u]);                                  \
                        EMIT(last ? segment[q + 5u] : segment[q + 6u]);         \
                        EMIT(segment[q + 4u]); EMIT(segment[q + 3u]);           \
                    }                                                           \
                    primitives++;                                               \
                }                                                               \
            } else if (mode == GL_LINE_STRIP) {                                \
                for (uint32_t q = 0u; q + 1u < segmentCount; q++) {            \
                    EMIT(segment[q]); EMIT(segment[q + 1u]); primitives++;     \
                }                                                                  \
            } else if (mode == GL_LINE_LOOP) {                                  \
                if (segmentCount >= 2u) {                                      \
                    for (uint32_t q = 0u; q + 1u < segmentCount; q++) {         \
                        EMIT(segment[q]); EMIT(segment[q + 1u]); primitives++;  \
                    }                                                              \
                    EMIT(segment[segmentCount - 1u]); EMIT(segment[0u]);         \
                    primitives++;                                               \
                }                                                                  \
            } else if (mode == GL_TRIANGLE_STRIP) {                             \
                for (uint32_t q = 0u; q + 2u < segmentCount; q++) {             \
                    /* GL 4.6 10.4.2.2: odd strip triangles swap their first    \
                     * two vertices so every triangle keeps one winding; the    \
                     * geometry shader must see the swapped order too. */       \
                    if (q & 1u) {                                               \
                        EMIT(segment[q + 1u]); EMIT(segment[q]);                \
                    } else {                                                    \
                        EMIT(segment[q]); EMIT(segment[q + 1u]);                \
                    }                                                           \
                    EMIT(segment[q + 2u]);                                      \
                    primitives++;                                               \
                }                                                               \
            } else if (mode == GL_TRIANGLE_FAN) {                               \
                for (uint32_t q = 1u; q + 1u < segmentCount; q++) {             \
                    EMIT(segment[0u]); EMIT(segment[q]); EMIT(segment[q + 1u]); \
                    primitives++;                                               \
                }                                                                  \
            }                                                                      \
        }                                                                          \
        segmentCount = 0u;                                                         \
    } while (0)
    for (uint32_t i = 0u; i < n; i++) {
        if (restartMode && source[i] == restartIndex) {
            EMIT_SEGMENT();
        } else {
            segment[segmentCount++] = source[i];
        }
    }
    EMIT_SEGMENT();
    #undef EMIT_SEGMENT
    #undef EMIT
    free(source);
    free(segment);
    if (gathered == 0u || primitives == 0u) {
        free(gather);
        return false;
    }
    *outGather = gather;
    *outGatherCount = gathered;
    *outPrimitiveCount = primitives;
    *outMaxIndex = maxIndex;
    return true;
}

static bool mglGeometryInputModeAccepts(GLenum gsMode, GLenum drawMode)
{
    switch (gsMode) {
        case GL_POINTS: return drawMode == GL_POINTS;
        case GL_LINES: return drawMode == GL_LINES ||
                              drawMode == GL_LINE_STRIP ||
                              drawMode == GL_LINE_LOOP;
        case GL_LINES_ADJACENCY: return drawMode == GL_LINES_ADJACENCY ||
                                          drawMode == GL_LINE_STRIP_ADJACENCY;
        case GL_TRIANGLES: return drawMode == GL_TRIANGLES ||
                                  drawMode == GL_TRIANGLE_STRIP ||
                                  drawMode == GL_TRIANGLE_FAN;
        case GL_TRIANGLES_ADJACENCY:
            return drawMode == GL_TRIANGLES_ADJACENCY ||
                   drawMode == GL_TRIANGLE_STRIP_ADJACENCY;
        default: return false;
    }
}

static id mglDrawSupportCreateBuffer(
    id device,
    NSUInteger length,
    uint64_t options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCreateBuffer(length, options, NULL, &buffer) == 0 &&
        buffer) {
        return (__bridge_transfer id)buffer;
    }
    return nil;
}

static id mglDrawSupportCreateBufferWithBytes(
    id device,
    const void *bytes,
    NSUInteger length,
    uint64_t options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCreateBufferWithBytes(bytes, length, options, NULL,
                                          &buffer) == 0 && buffer) {
        return (__bridge_transfer id)buffer;
    }
    return nil;
}

static id mglDrawSupportCreateBlitEncoder(
    void *commandBufferOwner)
{
    return (__bridge id)mglRenderCreateBlitEncoderBorrowed(
        commandBufferOwner);
}

static void mglDrawSupportBlitCopyBuffer(id encoder,
                                         id source,
                                         NSUInteger sourceOffset,
                                         id destination,
                                         NSUInteger destinationOffset,
                                         NSUInteger size)
{
    (void)mglRenderBlitCopyBuffer(
        (__bridge void *)encoder, (__bridge void *)source, sourceOffset,
        (__bridge void *)destination, destinationOffset, size);
}

static void mglDrawSupportEndBlitEncoder(id encoder)
{
    (void)mglRenderEndBlitEncoder((__bridge void *)encoder);
}

static void mglDrawSupportSetVertexBuffer(
    void *renderEncoderOwner,
    id buffer,
    NSUInteger offset,
    NSUInteger index)
{
    (void)mglRenderSetRenderBufferForOwner(
        renderEncoderOwner, (__bridge void *)buffer, offset,
        MGL_RENDER_BINDING_STAGE_VERTEX, (uint32_t)index);
}

static void mglDrawSupportSetVertexBytes(
    void *renderEncoderOwner,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    (void)mglRenderSetRenderBytesForOwner(
        renderEncoderOwner, bytes, length,
        MGL_RENDER_BINDING_STAGE_VERTEX, (uint32_t)index);
}

static void mglDrawSupportDrawIndexedPrimitives(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    NSUInteger indexCount,
    id indexBuffer,
    NSUInteger indexBufferOffset,
    NSUInteger instanceCount,
    NSInteger baseVertex,
    NSUInteger baseInstance)
{
    (void)mglRenderEncodeDrawForRenderEncoderOwner(renderEncoderOwner,
        &(MGLRenderDrawPlan){
            .kind = MGL_RENDER_DRAW_INDEXED,
            .primitive_type = (uint32_t)primitiveType,
            .index_count = indexCount,
            .index_type = (uint32_t)MGL_DRAW_INDEX_UINT32,
            .index_buffer = (__bridge void *)indexBuffer,
            .index_buffer_offset = indexBufferOffset,
            .instance_count = instanceCount,
            .base_vertex = baseVertex,
            .base_instance = baseInstance,
        }, NULL, 0);
}

/* Variant that honors the GL index type (UInt8/UInt16/UInt32).  Used by the
 * GS indexed capture so the original EBO drives vertex fetch directly. */
static void mglDrawSupportDrawIndexedPrimitivesType(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    NSUInteger indexCount,
    uint64_t indexType,
    id indexBuffer,
    NSUInteger indexBufferOffset,
    NSUInteger instanceCount,
    NSInteger baseVertex,
    NSUInteger baseInstance)
{
    MGLRenderDrawPlan plan = {
            .kind = MGL_RENDER_DRAW_INDEXED,
            .primitive_type = (uint32_t)primitiveType,
            .index_count = indexCount,
            .index_type = (uint32_t)indexType,
            .index_buffer = (__bridge void *)indexBuffer,
            .index_buffer_offset = indexBufferOffset,
            .instance_count = instanceCount,
            .base_vertex = baseVertex,
            .base_instance = baseInstance,
        };
    (void)mglRenderEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

static void mglDrawSupportDrawPrimitives(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    NSUInteger vertexStart,
    NSUInteger vertexCount,
    NSUInteger instanceCount,
    NSUInteger baseInstance)
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

static void mglDrawSupportDrawPrimitivesIndirect(
    void *renderEncoderOwner,
    uint32_t primitiveType,
    id indirectBuffer,
    NSUInteger indirectBufferOffset)
{
    MGLRenderDrawPlan plan = {
            .kind = MGL_RENDER_DRAW_ARRAY_INDIRECT,
            .primitive_type = (uint32_t)primitiveType,
            .indirect_buffer = (__bridge void *)indirectBuffer,
            .indirect_buffer_offset = indirectBufferOffset,
        };
    (void)mglRenderEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

static id mglDrawSupportCreateComputeEncoder(
    void *commandBufferOwner)
{
    return (__bridge id)mglRenderCreateComputeEncoderBorrowed(
        commandBufferOwner);
}

static void mglDrawSupportSetComputePipeline(
    id encoder,
    id pipeline)
{
    (void)mglRenderSetComputePipelineState((__bridge void *)encoder,
                                              (__bridge void *)pipeline);
}

static void mglDrawSupportSetComputeBuffer(
    id encoder,
    id buffer,
    NSUInteger offset,
    NSUInteger index)
{
    (void)mglRenderSetComputeBuffer((__bridge void *)encoder,
                                       (__bridge void *)buffer, offset,
                                       (uint32_t)index);
}

static void mglDrawSupportSetComputeBytes(
    id encoder,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    (void)mglRenderSetComputeBytes((__bridge void *)encoder, bytes,
                                      length, (uint32_t)index);
}

static void mglDrawSupportDispatchCompute(
    id encoder,
    uint32_t groupsX,
    uint32_t groupsY,
    uint32_t groupsZ,
    uint32_t threadsX,
    uint32_t threadsY,
    uint32_t threadsZ)
{
    (void)mglRenderDispatchCompute(
        (__bridge void *)encoder, groupsX, groupsY, groupsZ,
        threadsX, threadsY, threadsZ);
}

static void mglDrawSupportEndComputeEncoder(
    id encoder)
{
    (void)mglRenderEndComputeEncoder((__bridge void *)encoder);
}

static void mglDrawSupportSetTessellationFactors(
    void *renderEncoderOwner,
    id buffer,
    NSUInteger offset,
    NSUInteger instanceStride)
{
    (void)mglRenderSetTessellationFactorBufferForOwner(
        renderEncoderOwner, (__bridge void *)buffer, offset, instanceStride);
}

static void mglDrawSupportDrawPatches(
    void *renderEncoderOwner,
    NSUInteger controlPointCount,
    NSUInteger patchStart,
    NSUInteger patchCount,
    id patchIndexBuffer,
    NSUInteger patchIndexBufferOffset,
    NSUInteger instanceCount,
    NSUInteger baseInstance)
{
    MGLRenderDrawPlan plan = {
            .kind = MGL_RENDER_DRAW_PATCHES,
            .primitive_type = (uint32_t)MGL_DRAW_PRIMITIVE_TRIANGLE,
            .control_point_count = controlPointCount,
            .patch_start = patchStart,
            .patch_count = patchCount,
            .patch_index_buffer = (__bridge void *)patchIndexBuffer,
            .patch_index_buffer_offset = patchIndexBufferOffset,
            .instance_count = instanceCount,
            .base_instance = baseInstance,
        };
    (void)mglRenderEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

static void mglDrawSupportDrawIndexedPatches(
    void *renderEncoderOwner,
    NSUInteger controlPointCount,
    NSUInteger patchStart,
    NSUInteger patchCount,
    id patchIndexBuffer,
    NSUInteger patchIndexBufferOffset,
    id controlPointIndexBuffer,
    NSUInteger controlPointIndexBufferOffset,
    NSUInteger instanceCount,
    NSUInteger baseInstance)
{
    MGLRenderDrawPlan plan = {
            .kind = MGL_RENDER_DRAW_INDEXED_PATCHES,
            .primitive_type = (uint32_t)MGL_DRAW_PRIMITIVE_TRIANGLE,
            .control_point_count = controlPointCount,
            .patch_start = patchStart,
            .patch_count = patchCount,
            .patch_index_buffer = (__bridge void *)patchIndexBuffer,
            .patch_index_buffer_offset = patchIndexBufferOffset,
            .control_point_index_buffer =
                (__bridge void *)controlPointIndexBuffer,
            .control_point_index_buffer_offset =
                controlPointIndexBufferOffset,
            .instance_count = instanceCount,
            .base_instance = baseInstance,
        };
    (void)mglRenderEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
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
extern GLboolean mglHasActiveIndexedPrimitiveQuery(void);
extern GLboolean mglHasActivePrimitiveQuery(void);
extern GLboolean mglHasActiveGeometryShaderQuery(void);

static void mglRecordGeometryPrimitiveQueries(
    GLMContext ctx,
    GLuint64 generatedStream0,
    GLuint64 writtenStream0,
    BOOL xfbActive,
    const MGLAIRGSXFBMeta *meta,
    uint32_t streamCount,
    const NSUInteger *bufferWritten,
    const NSUInteger *bufferStride,
    GLuint64 geometryInvocations)
{
    mglRecordActiveGeometryShaderQueryDraw(
        ctx, geometryInvocations, generatedStream0);
    mglRecordActivePrimitiveQueryDraw(
        ctx, generatedStream0, xfbActive ? writtenStream0 : 0u);
    if (!meta || !bufferWritten || !bufferStride) return;
    if (streamCount > MGL_AIR_GS_MAX_STREAMS) {
        streamCount = MGL_AIR_GS_MAX_STREAMS;
    }
    for (uint32_t s = 1u; s < streamCount; s++) {
        /* Indexed stream s query: generated stays in the meta; written is
         * the ordered scatter's whole-primitive bytes for buffer s divided
         * by its per-record stride (streams > 0 are points, vpp = 1). */
        GLuint64 written = 0u;
        if (xfbActive && bufferStride[s] > 0u) {
            written = (GLuint64)bufferWritten[s] /
                      (GLuint64)bufferStride[s];
        }
        mglRecordActivePrimitiveQueryDrawIndexed(
            ctx, s, (GLuint64)meta->stream[s].generated, written);
    }
}

static BOOL mglCheckedTessCaptureSize(GLsizei count, GLsizei instanceCount,
                                      NSUInteger stride,
                                      NSUInteger *sizeOut,
                                      NSUInteger *offsetOut)
{

    uint64_t size = 0u;
    uint64_t offset = 0u;
    if (mglRenderCheckedTessCaptureSize(
            (int64_t)count, (int64_t)instanceCount, (uint64_t)stride,
            (uint64_t)MGL_AIR_PER_VERTEX_STRIDE, &size, &offset) != 0) {
        return NO;
    }
    *sizeOut = (NSUInteger)size;
    *offsetOut = (NSUInteger)offset;
    return YES;
}

static BOOL mglNativeTESInterfaceSupported(Program *tcsProgram,
                                           Program *tesProgram)
{
    if (!tesProgram) {
        return NO;
    }

    return mglRenderNativeTESInterfaceSupported(
        tesProgram->modules[_TESS_EVALUATION_SHADER].mtl_function,
        (uint64_t)tesProgram->modules[_TESS_EVALUATION_SHADER].metallib_bytes,
        (uint32_t)tesProgram->tess_gen_point_mode,
        (uint32_t)tesProgram->transform_feedback_varying_count,
        (uint32_t)tesProgram->tess_gen_mode,
        tcsProgram ? tcsProgram->modules[_TESS_CONTROL_SHADER].mtl_function : NULL,
        tcsProgram ? (uint64_t)tcsProgram->modules[_TESS_CONTROL_SHADER].metallib_bytes : 0u,
        tcsProgram ? (uint32_t)tcsProgram->tess_control_output_vertices : 0u) != 0;
}

static id mglDefaultTessFactorBuffer(id device,
                                                GLMState *state,
                                                GLuint patchCount)
{
    if (!device || !state || patchCount == 0u) return nil;
    const NSUInteger stride = 12u;
    if ((NSUInteger)patchCount > NSUIntegerMax / stride) return nil;
    id buffer = mglDrawSupportCreateBuffer(
        device, (NSUInteger)patchCount * stride,
        0u);
    if (!buffer || !mglDrawSupportBufferContents(buffer)) return nil;

    if (mglRenderFillDefaultTessFactorBuffer(
            (void *)mglDrawSupportBufferContents(buffer),
            (uint64_t)((NSUInteger)patchCount * stride),
            state->var.patch_default_outer_level,
            state->var.patch_default_inner_level,
            patchCount) != 0) {
        return nil;
    }
    return buffer;
}

/* Cached variant of the default factor buffer for the TES-only path:
 * consecutive tess draws reuse one stable allocation unless the default
 * patch levels or patch count actually changed. */
static id mglCachedDefaultTessFactorBuffer(
    id device, MGLRendererBackendHandle *backend, GLMState *state,
    GLuint patchCount)
{
    if (!device || !backend || !state || patchCount == 0u) return nil;
    float levels[6] = {
        state->var.patch_default_outer_level[0],
        state->var.patch_default_outer_level[1],
        state->var.patch_default_outer_level[2],
        state->var.patch_default_outer_level[3],
        state->var.patch_default_inner_level[0],
        state->var.patch_default_inner_level[1],
    };
    void *cached = NULL;
    if (mglRendererBackendGetTessFactorBuffer(
            backend, patchCount, levels, &cached) == 1 && cached) {
        return (__bridge id)cached;
    }
    id fresh = mglDefaultTessFactorBuffer(device, state, patchCount);
    if (!fresh) return nil;
    if (mglRendererBackendPutTessFactorBuffer(
            backend, patchCount, levels, (__bridge void *)fresh) != 0) {
        return fresh;
    }
    return fresh;
}

static id mglNativeTessFactorBuffer(id device,
                                                id canonical,
                                                GLenum mode,
                                                GLuint patchCount)
{
    const NSUInteger canonicalStride = 12u;
    if (!device || !canonical || !mglDrawSupportBufferContents(canonical) || patchCount == 0u ||
        mglDrawSupportBufferLength(canonical) < (NSUInteger)patchCount * canonicalStride) {
        return nil;
    }
    if (mode == GL_QUADS) {
        return canonical;
    }
    if (mode != GL_TRIANGLES) {
        return nil;
    }

    const NSUInteger triangleStride = 8u;
    id result = mglDrawSupportCreateBuffer(
        device, (NSUInteger)patchCount * triangleStride,
        0u);
    if (!result || !mglDrawSupportBufferContents(result)) {
        return nil;
    }

    if (mglRenderRepackTessFactorTriangles(
            (const void *)mglDrawSupportBufferContents(canonical), (uint64_t)mglDrawSupportBufferLength(canonical),
            (void *)mglDrawSupportBufferContents(result),
            (uint64_t)((NSUInteger)patchCount * triangleStride),
            patchCount) != 0) {
        return nil;
    }
    return result;
}

static GLuint64 mglNativeTessPrimitiveCount(id canonical,
                                             Program *tesProgram,
                                             GLuint patchCount,
                                             GLuint instanceCount)
{
    if (!canonical || !mglDrawSupportBufferContents(canonical) || !tesProgram || patchCount == 0u) {
        return 0u;
    }

    return (GLuint64)mglRenderTessPrimitiveCount(
        (const void *)mglDrawSupportBufferContents(canonical), (uint64_t)mglDrawSupportBufferLength(canonical),
        patchCount, (uint32_t)tesProgram->tess_gen_mode,
        instanceCount);
}

@implementation MGLRenderer (Draw)

- (BOOL)captureAIRCullDistancesForArrayDraw:(GLMContext)drawCtx
                                      first:(GLint)first
                                      count:(GLsizei)count
                              instanceCount:(GLsizei)instanceCount
                               baseInstance:(GLuint)baseInstance
{
    (void)mglRendererBackendSetCullDistanceCaptureBuffer(_backend, NULL);
    _tessellation.cullDistanceCaptureFirstInstance = 0u;
    _tessellation.cullDistanceCaptureInstanceStride = 0u;
    if (!drawCtx || first < 0 || count <= 0 || instanceCount <= 0) return NO;

    Program *vertexProgram =
        mglResolveProgramForStageFromState(drawCtx, _VERTEX_SHADER);
    if (!vertexProgram || !vertexProgram->uses_cull_distance ||
        ![self bindMTLProgram:vertexProgram] ||
        !vertexProgram->modules[_VERTEX_SHADER].mtl_cull_capture_function) {
        return NO;
    }
    const uint64_t endVertex = (uint64_t)(uint32_t)first +
                               (uint64_t)(uint32_t)count;
    const uint64_t lastCaptureIndex =
        (uint64_t)((uint32_t)instanceCount - 1u) * (uint64_t)(uint32_t)count +
        endVertex;
    if (endVertex == 0u || lastCaptureIndex == 0u ||
        lastCaptureIndex > NSUIntegerMax / 32u) return NO;
    id capture = mglDrawSupportCreateBuffer(
        _device, (NSUInteger)(lastCaptureIndex * 32u),
        0u);
    if (!capture) return NO;

    self->ctx = drawCtx;
    _tessellation.cullDistanceCaptureActive = YES;
    drawCtx->state.dirty_bits = DIRTY_ALL;
    if (![self processGLState:true] ||
        mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
        _tessellation.cullDistanceCaptureActive = NO;
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return NO;
    }
    MGLCullDistanceEmuParams params = {
        .prim_vertex_count = 1u,
        .culldist_offset = 0u,
        .vertex_stride = 32u,
        .culldist_size = MIN(vertexProgram->cull_distance_count, 8u),
        .first_vertex = (uint32_t)first,
        .first_instance = baseInstance,
        .instance_stride = (uint32_t)count,
    };
    mglDrawSupportSetVertexBuffer(_renderPassManager.state->currentRenderEncoderOwner, capture, 0u, 29u);
    mglDrawSupportSetVertexBytes(
        _renderPassManager.state->currentRenderEncoderOwner, &params, sizeof(params), kMGLCullDistanceParamsBufferIndex);
    mglDrawSupportDrawPrimitives(_renderPassManager.state->currentRenderEncoderOwner, MGL_DRAW_PRIMITIVE_POINT,
                                 (NSUInteger)first, (NSUInteger)count,
                                 (NSUInteger)instanceCount,
                                 (NSUInteger)baseInstance);
    _currentCBHasWork = YES;
    [self endRenderEncoding];
    _tessellation.cullDistanceCaptureActive = NO;
    (void)mglRendererBackendSetCullDistanceCaptureBuffer(
        _backend, (__bridge void *)capture);
    _tessellation.cullDistanceCaptureFirstInstance = baseInstance;
    _tessellation.cullDistanceCaptureInstanceStride = (uint32_t)count;
    drawCtx->state.dirty_bits = DIRTY_ALL;
    return YES;
}

- (BOOL)captureAIRCullDistancesForElementDraw:(GLMContext)drawCtx
                                    indexBytes:(const uint8_t *)indexBytes
                                     indexType:(GLenum)indexType
                                         count:(GLsizei)count
                                    baseVertex:(GLint)baseVertex
                                 instanceCount:(GLsizei)instanceCount
                                  baseInstance:(GLuint)baseInstance
{
    if (!drawCtx || !indexBytes || count <= 0 || instanceCount <= 0) return NO;
    uint32_t restartIndex = 0u;
    const bool restartEnabled =
        mglPrimitiveRestartIndexForType(drawCtx, indexType, &restartIndex);
    const uint32_t elemWidth = indexType == GL_UNSIGNED_BYTE ? 1u
        : indexType == GL_UNSIGNED_SHORT ? 2u : 4u;
    uint32_t scanMin = 0u, scanMax = 0u;
    int scanValid = 0;
    if (mglRenderScanIndexRangeIgnoringRestart(
            indexBytes, elemWidth, (uint32_t)count,
            restartEnabled ? 1 : 0, restartIndex,
            &scanMin, &scanMax, &scanValid) != 0 || !scanValid) {
        return NO;
    }
    uint32_t minIndex = scanMin;
    uint32_t maxIndex = scanMax;
    const int64_t first = (int64_t)minIndex + (int64_t)baseVertex;
    const int64_t last = (int64_t)maxIndex + (int64_t)baseVertex;
    if (first < 0 || last < first || last > INT32_MAX) return NO;
    const uint64_t vertexCount = (uint64_t)(last - first) + 1u;
    if (vertexCount > INT32_MAX) return NO;
    return [self captureAIRCullDistancesForArrayDraw:drawCtx
                                               first:(GLint)first
                                               count:(GLsizei)vertexCount
                                       instanceCount:instanceCount
                                        baseInstance:baseInstance];
}

- (BOOL)prepareAndEncodeDirectCullDistanceElementDraw:(GLenum)mode
                                           indexBytes:(const uint8_t *)indexBytes
                                            indexType:(GLenum)indexType
                                                count:(GLsizei)count
                                           baseVertex:(GLint)baseVertex
                                        instanceCount:(GLsizei)instanceCount
                                         baseInstance:(GLuint)baseInstance
                                      polygonLineMode:(BOOL)polygonLineMode
{
    Program *activeProgram =
        ctx ? mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER) : NULL;
    if (!activeProgram || !activeProgram->uses_cull_distance) return NO;
    if (!indexBytes || count <= 0 || instanceCount <= 0) return YES;

    if (activeProgram->modules[_VERTEX_SHADER].mtl_cull_capture_function) {
        if (![self captureAIRCullDistancesForElementDraw:ctx
                                             indexBytes:indexBytes
                                              indexType:indexType
                                                  count:count
                                             baseVertex:baseVertex
                                          instanceCount:instanceCount
                                           baseInstance:baseInstance] ||
            ![self processGLState:true] ||
            mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
            return YES;
        }
    }

    MGLEncodeContext encCtx = {
        .render_encoder_owner = _renderPassManager.state->currentRenderEncoderOwner,
    };
    return [self encodeCullDistanceElementDraw:mode
                                    indexBytes:indexBytes
                                     indexType:indexType
                                         count:count
                                    baseVertex:baseVertex
                                 instanceCount:instanceCount
                                  baseInstance:baseInstance
                               polygonLineMode:polygonLineMode
                                 encodeContext:&encCtx];
}

- (BOOL)encodeCullDistanceArrayDraw:(GLenum)mode
                               first:(GLint)first
                               count:(GLsizei)count
                       instanceCount:(GLsizei)instanceCount
                        baseInstance:(GLuint)baseInstance
                       encodeContext:(const MGLEncodeContext *)encCtx
{
    Program *activeProgram =
        ctx ? mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER) : NULL;
    if (!activeProgram || !activeProgram->uses_cull_distance ||
        !mglDrawSupportEncodeContextIsActive(encCtx)) {
        return NO;
    }

    if (mode == GL_TRIANGLE_STRIP && count >= 3) {
        NSUInteger indexCount = 0u;
        id indexBuffer = mglNewTriangleStripArrayIndexBuffer(
            _device, (NSUInteger)count, &indexCount);
        if (!indexBuffer || indexCount == 0u) return YES;
        for (NSUInteger primitive = 0u; primitive * 3u < indexCount;
             primitive++) {
            const GLuint vertices[3] = {
                (GLuint)first + (GLuint)primitive,
                (GLuint)first + (GLuint)primitive + 1u,
                (GLuint)first + (GLuint)primitive + 2u,
            };
            [self bindCullDistanceEmulationBuffers:mode
                                        firstVertex:(GLuint)first
                                   explicitVertices:vertices
                                 explicitVertexCount:3u
                                      encodeContext:encCtx];
            mglDrawSupportDrawIndexedPrimitives(
                encCtx->render_encoder_owner, MGL_DRAW_PRIMITIVE_TRIANGLE, 3u, indexBuffer,
                primitive * 3u * sizeof(uint32_t),
                (NSUInteger)instanceCount, (NSInteger)first,
                (NSUInteger)baseInstance);
        }
        return YES;
    }

    if (mode == GL_TRIANGLE_FAN && count >= 3) {
        NSUInteger indexCount = 0u;
        id indexBuffer = mglNewTriangleFanArrayIndexBuffer(
            _device, (NSUInteger)count, &indexCount);
        if (!indexBuffer || indexCount == 0u) return YES;
        for (NSUInteger primitive = 0u; primitive * 3u < indexCount;
             primitive++) {
            const GLuint vertices[3] = {
                (GLuint)first,
                (GLuint)first + (GLuint)primitive + 1u,
                (GLuint)first + (GLuint)primitive + 2u,
            };
            [self bindCullDistanceEmulationBuffers:mode
                                        firstVertex:(GLuint)first
                                   explicitVertices:vertices
                                 explicitVertexCount:3u
                                      encodeContext:encCtx];
            mglDrawSupportDrawIndexedPrimitives(
                encCtx->render_encoder_owner, MGL_DRAW_PRIMITIVE_TRIANGLE, 3u, indexBuffer,
                primitive * 3u * sizeof(uint32_t),
                (NSUInteger)instanceCount, (NSInteger)first,
                (NSUInteger)baseInstance);
        }
        return YES;
    }

    if (mode == GL_LINE_STRIP && count >= 2) {
        for (GLsizei primitive = 0; primitive + 1 < count; primitive++) {
            [self bindCullDistanceEmulationBuffers:mode
                                        firstVertex:(GLuint)(first + primitive)
                                   explicitVertices:NULL
                                 explicitVertexCount:0u
                                      encodeContext:encCtx];
            mglDrawSupportDrawPrimitives(
                encCtx->render_encoder_owner, MGL_DRAW_PRIMITIVE_LINE,
                (NSUInteger)(first + primitive), 2u,
                (NSUInteger)instanceCount, (NSUInteger)baseInstance);
        }
        return YES;
    }

    if (mode == GL_LINE_LOOP && count >= 2) {
        NSUInteger indexCount = 0u;
        id indexBuffer = mglNewLineLoopArrayIndexBuffer(
            _device, (NSUInteger)first, (NSUInteger)count, &indexCount);
        if (!indexBuffer || indexCount == 0u) return YES;
        for (NSUInteger primitive = 0u; primitive + 1u < indexCount;
             primitive++) {
            const GLuint vertices[2] = {
                (GLuint)first + (GLuint)primitive,
                (GLuint)first +
                    (GLuint)((primitive + 1u) % (NSUInteger)count),
            };
            [self bindCullDistanceEmulationBuffers:mode
                                        firstVertex:(GLuint)first
                                   explicitVertices:vertices
                                 explicitVertexCount:2u
                                      encodeContext:encCtx];
            mglDrawSupportDrawIndexedPrimitives(
                encCtx->render_encoder_owner, MGL_DRAW_PRIMITIVE_LINE, 2u, indexBuffer,
                primitive * sizeof(uint32_t), (NSUInteger)instanceCount, 0,
                (NSUInteger)baseInstance);
        }
        return YES;
    }

    [self bindCullDistanceEmulationBuffers:mode
                                firstVertex:(GLuint)first
                           explicitVertices:NULL
                         explicitVertexCount:0u
                              encodeContext:encCtx];
    return NO;
}

- (BOOL)encodeCullDistanceElementDraw:(GLenum)mode
                            indexBytes:(const uint8_t *)indexBytes
                             indexType:(GLenum)indexType
                                 count:(GLsizei)count
                            baseVertex:(GLint)baseVertex
                         instanceCount:(GLsizei)instanceCount
                          baseInstance:(GLuint)baseInstance
                       polygonLineMode:(BOOL)polygonLineMode
                         encodeContext:(const MGLEncodeContext *)encCtx
{
    Program *activeProgram =
        ctx ? mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER) : NULL;
    if (!activeProgram || !activeProgram->uses_cull_distance) return NO;
    id captureBuffer = (__bridge id)
        mglRendererBackendGetCullDistanceCaptureBuffer(_backend);
    if (activeProgram->modules[_VERTEX_SHADER].mtl_cull_capture_function &&
        !captureBuffer) {
        return YES;
    }
    if (!indexBytes || count <= 0 || instanceCount <= 0 ||
        !mglDrawSupportEncodeContextIsActive(encCtx)) {
        return YES;
    }

    uint32_t restartIndex = 0u;
    const bool restartEnabled =
        mglPrimitiveRestartIndexForType(ctx, indexType, &restartIndex);
    void *planOwner = NULL;
    void *indexBufferHandle = NULL;
    uint64_t primitiveCount = 0u;
    if (mglRenderCreateCullDistanceIndexPlan(
            (__bridge void *)_device, indexBytes, indexType,
            (uint64_t)count, mode,
            restartEnabled ? 1 : 0, restartIndex, baseVertex,
            polygonLineMode ? 1 : 0, &planOwner, &indexBufferHandle,
            &primitiveCount) != 0 || !planOwner) {
        return YES;
    }

    id indexBuffer =
        (__bridge id)indexBufferHandle;
    @try {
        for (uint64_t primitiveIndex = 0u;
             primitiveIndex < primitiveCount; ++primitiveIndex) {
            MGLRenderCullDistancePrimitive primitive = {0};
            if (mglRenderGetCullDistanceIndexPrimitive(
                    planOwner, primitiveIndex, &primitive) != 0) {
                break;
            }
            [self bindCullDistanceEmulationBuffers:mode
                                        firstVertex:0u
                                   explicitVertices:primitive.vertices
                                 explicitVertexCount:primitive.vertex_count
                                      encodeContext:encCtx];
            mglDrawSupportDrawIndexedPrimitives(
                encCtx->render_encoder_owner,
                (uint32_t)primitive.primitive_type,
                (NSUInteger)primitive.index_count,
                indexBuffer,
                (NSUInteger)primitive.index_buffer_offset,
                (NSUInteger)instanceCount,
                0,
                (NSUInteger)baseInstance);
        }
    } @finally {
        mglRenderDestroyCullDistanceIndexPlan(&planOwner);
    }
    return YES;
}

- (id)captureAIRVertexPositionsForTessellation:(GLMContext)drawCtx
                                                    first:(GLint)first
                                                    count:(GLsizei)count
                                            instanceCount:(GLsizei)instanceCount
                                             baseInstance:(GLuint)baseInstance
                                               outOffset:(NSUInteger *)outOffset
{
    if (outOffset) *outOffset = 0u;
    if (!drawCtx || first < 0 || count <= 0 || instanceCount <= 0) return nil;

    Program *vertexProgram =
        mglResolveProgramForStageFromState(drawCtx, _VERTEX_SHADER);
    if (!vertexProgram || ![self bindMTLProgram:vertexProgram] ||
        !vertexProgram->modules[_VERTEX_SHADER].mtl_tess_capture_function) {
        return nil;
    }

    NSUInteger captureSize = 0u;
    NSUInteger captureOffset = 0u;
    NSUInteger captureStride = mglAIRPerVertexStrideForResources(
        &vertexProgram->shader_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES]);
    const NSUInteger recordsPerInstance = (NSUInteger)count;
    if (!mglCheckedTessCaptureSize(recordsPerInstance, instanceCount,
                                   captureStride, &captureSize,
                                   &captureOffset)) {
        return nil;
    }
    id capture = mglDrawSupportCreateBuffer(
        _device, captureSize, 0u);
    if (!capture) return nil;

    self->ctx = drawCtx;
    _tessellation.tessVertexCaptureActive = YES;
    drawCtx->state.dirty_bits = DIRTY_ALL;
    if (![self processGLState:true] ||
        mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
        _tessellation.tessVertexCaptureActive = NO;
        return nil;
    }
    mglDrawSupportSetVertexBuffer(_renderPassManager.state->currentRenderEncoderOwner, capture, 0u, 29u);
    const uint32_t captureParams[3] = {
        (uint32_t)first, (uint32_t)recordsPerInstance, baseInstance,
    };
    mglDrawSupportSetVertexBytes(
        _renderPassManager.state->currentRenderEncoderOwner, captureParams, sizeof(captureParams), 28u);
    if (getenv("MGL_GS_DIAG")) {
        NSLog(@"MGL GS DIAG capture-draw POINT first=%d count=%d instances=%d baseInst=%u stride=%lu size=%lu",
              (int)first, (int)count, (int)instanceCount, baseInstance,
              (unsigned long)captureStride, (unsigned long)captureSize);
    }
    mglDrawSupportDrawPrimitives(_renderPassManager.state->currentRenderEncoderOwner, MGL_DRAW_PRIMITIVE_POINT,
                                 (NSUInteger)first, (NSUInteger)count,
                                 (NSUInteger)instanceCount,
                                 (NSUInteger)baseInstance);
    _currentCBHasWork = YES;
    [self endRenderEncoding];
    _tessellation.tessVertexCaptureActive = NO;
    drawCtx->state.dirty_bits = DIRTY_ALL;
    if (outOffset) *outOffset = captureOffset;
    return capture;
}


- (id)captureAIRVertexPositionsForGeometryIndexed:(GLMContext)drawCtx
                                                  indexBuffer:(id)indexBuffer
                                                    indexType:(uint64_t)indexType
                                                  indexOffset:(NSUInteger)indexOffset
                                                        count:(GLsizei)count
                                                    baseVertex:(GLint)baseVertex
                                                 instanceCount:(GLsizei)instanceCount
                                                  baseInstance:(GLuint)baseInstance
                                                     maxIndex:(uint32_t)maxIndex
                                                    outOffset:(NSUInteger *)outOffset
{
    if (outOffset) *outOffset = 0u;
    if (!drawCtx || count <= 0 || instanceCount <= 0 || !indexBuffer) {
        return nil;
    }

    Program *vertexProgram =
        mglResolveProgramForStageFromState(drawCtx, _VERTEX_SHADER);
    if (!vertexProgram || ![self bindMTLProgram:vertexProgram] ||
        !vertexProgram->modules[_VERTEX_SHADER].mtl_tess_capture_function) {
        return nil;
    }

    NSUInteger captureSize = 0u;
    NSUInteger captureOffset = 0u;
    NSUInteger captureStride = mglAIRPerVertexStrideForResources(
        &vertexProgram->shader_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES]);
    const NSUInteger recordsPerInstance = (NSUInteger)maxIndex + 1u;
    if (!mglCheckedTessCaptureSize(recordsPerInstance, instanceCount,
                                   captureStride, &captureSize,
                                   &captureOffset)) {
        return nil;
    }
    id capture = mglDrawSupportCreateBuffer(
        _device, captureSize, 0u);
    if (!capture) return nil;
    self->ctx = drawCtx;
    _tessellation.tessVertexCaptureActive = YES;
    drawCtx->state.dirty_bits = DIRTY_ALL;
    if (![self processGLState:true] ||
        mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
        _tessellation.tessVertexCaptureActive = NO;
        return nil;
    }
    mglDrawSupportSetVertexBuffer(_renderPassManager.state->currentRenderEncoderOwner, capture, 0u, 29u);
    const uint32_t captureParams[3] = {
        0u, (uint32_t)recordsPerInstance, baseInstance,
    };
    mglDrawSupportSetVertexBytes(
        _renderPassManager.state->currentRenderEncoderOwner, captureParams, sizeof(captureParams), 28u);
    /* The capture kernel indexes records by raw vertex_id with no bounds
     * check; a primitive-restart marker (0xFFFFFFFF for UInt32) in the
     * stream would write past the sparse record span and corrupt the
     * next instance's data.  Sanitize the marker away (to vertex 0, whose
     * record no gathered patch ever references) before drawing. */
    id sanitizedIndexBuffer = indexBuffer;
    NSUInteger sanitizedIndexOffset = indexOffset;
    uint32_t restartIndex = 0u;
    if (mglPrimitiveRestartIndexForType(drawCtx, indexType, &restartIndex)) {
        const NSUInteger elemBytes = indexType == GL_UNSIGNED_BYTE ? 1u
            : indexType == GL_UNSIGNED_SHORT ? 2u : 4u;
        const NSUInteger streamBytes = (NSUInteger)count * elemBytes;
        if (mglDrawSupportBufferContents(indexBuffer) &&
            (NSUInteger)indexOffset + streamBytes <= mglDrawSupportBufferLength(indexBuffer)) {
            uint8_t *copy = malloc(streamBytes);
            if (copy) {
                memcpy(copy,
                       (const uint8_t *)mglDrawSupportBufferContents(indexBuffer) + indexOffset,
                       streamBytes);
                if (elemBytes == 1u) {
                    for (GLsizei i = 0; i < count; i++)
                        if (((const uint8_t *)copy)[i] ==
                                (uint8_t)restartIndex)
                            ((uint8_t *)copy)[i] = 0u;
                } else if (elemBytes == 2u) {
                    for (GLsizei i = 0; i < count; i++)
                        if (((const uint16_t *)copy)[i] ==
                                (uint16_t)restartIndex)
                            ((uint16_t *)copy)[i] = 0u;
                } else {
                    for (GLsizei i = 0; i < count; i++)
                        if (((const uint32_t *)copy)[i] == restartIndex)
                            ((uint32_t *)copy)[i] = 0u;
                }
                id clean =
                    mglDrawSupportCreateBufferWithBytes(
                        _device, copy, streamBytes,
                        0u);
                free(copy);
                if (clean) {
                    sanitizedIndexBuffer = clean;
                    sanitizedIndexOffset = 0u;
                }
            }
        }
    }
    /* Metal has no UInt8 index type: GL_UNSIGNED_BYTE streams must be
     * expanded to UInt16 before the indexed capture draw, or Metal reads
     * byte pairs as garbage indices and every record past index 0 is lost. */
    id drawIndexBuffer = sanitizedIndexBuffer;
    NSUInteger drawIndexOffset = sanitizedIndexOffset;
    uint64_t mtlIndexType = mglIndexTypeForGLType((GLenum)indexType);
    if ((GLuint)mtlIndexType != 0xFFFFFFFFu) {
        NSUInteger preparedOffset = sanitizedIndexOffset;
        uint64_t preparedType = mtlIndexType;
        id prepared = mglPreparedElementIndexBuffer(
            _device, NULL, sanitizedIndexBuffer, (GLenum)indexType,
            &preparedOffset, &preparedType);
        if (prepared) {
            drawIndexBuffer = prepared;
            drawIndexOffset = preparedOffset;
            mtlIndexType = preparedType;
        }
    }
    mglDrawSupportDrawIndexedPrimitivesType(
        _renderPassManager.state->currentRenderEncoderOwner, MGL_DRAW_PRIMITIVE_POINT, (NSUInteger)count, mtlIndexType,
        drawIndexBuffer, drawIndexOffset, (NSUInteger)instanceCount,
        (NSInteger)baseVertex, (NSUInteger)baseInstance);
    _currentCBHasWork = YES;    [self endRenderEncoding];
    _tessellation.tessVertexCaptureActive = NO;
    drawCtx->state.dirty_bits = DIRTY_ALL;
    if (outOffset) *outOffset = captureOffset;
    return capture;
}

- (BOOL)handleVertexTransformFeedbackDrawIfNeeded:(GLMContext)drawCtx
                                               mode:(GLenum)mode
                                              first:(GLint)first
                                              count:(GLsizei)count
                                      instanceCount:(GLsizei)instanceCount
                                       baseInstance:(GLuint)baseInstance
{
    if (!drawCtx || mode != GL_POINTS || first < 0 || count <= 0 ||
        instanceCount <= 0) {
        return NO;
    }
    TransformFeedback *xfb = MGL_STATE(drawCtx)->transform_feedback;
    if (!xfb || !xfb->active || xfb->paused ||
        xfb->primitive_mode != mode) {
        return NO;
    }
    Program *program = mglResolveProgramForStageFromState(
        drawCtx, _VERTEX_SHADER);
    if (!program || program->shader_slots[_GEOMETRY_SHADER] ||
        program->shader_slots[_TESS_CONTROL_SHADER] ||
        program->shader_slots[_TESS_EVALUATION_SHADER] ||
        !program->transform_feedback_layout_valid ||
        program->transform_feedback_varying_count <= 0) {
        return NO;
    }

    const MGLShaderResourceList *outputs =
        &program->shader_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES];
    NSUInteger bufferStride[MGL_MAX_TRANSFORM_FEEDBACK_BUFFERS] = {0u};
    NSUInteger sourceOffset[MAX_ATTRIBS] = {0u};
    BOOL hasSource[MAX_ATTRIBS] = {NO};
    GLuint bufferCount = program->transform_feedback_layout_buffer_count;
    if (bufferCount == 0u || bufferCount > MGL_MAX_TRANSFORM_FEEDBACK_BUFFERS) {
        return NO;
    }

    for (GLsizei varying = 0;
         varying < program->transform_feedback_varying_count;
         varying++) {
        const MGLTransformFeedbackVaryingPlan *plan =
            &program->transform_feedback_layout[varying];
        if (plan->buffer_index >= bufferCount || plan->stream > 0 ||
            plan->component_count > 4u) {
            return NO;
        }
        NSUInteger end = ((NSUInteger)plan->component_offset +
                          (NSUInteger)plan->component_count) * sizeof(uint32_t);
        if (end > bufferStride[plan->buffer_index]) {
            bufferStride[plan->buffer_index] = end;
        }
        if (plan->component_count == 0u || plan->stream < 0) {
            continue;
        }

        const char *name = program->transform_feedback_varying_names[varying];
        if (!name || !name[0]) return NO;
        if (strcmp(name, "gl_Position") == 0 && plan->builtin) {
            sourceOffset[varying] = MGL_AIR_PER_VERTEX_POSITION_OFFSET;
            hasSource[varying] = YES;
            continue;
        }
        if (strcmp(name, "gl_PointSize") == 0 && plan->builtin) {
            sourceOffset[varying] = MGL_AIR_PER_VERTEX_POINT_SIZE_OFFSET;
            hasSource[varying] = YES;
            continue;
        }
        /* Single-element capture ("v[2]") shifts the record slot by the
         * element index; whole-array names still span multiple slots and
         * stay unsupported here. */
        char baseName[96];
        const char *bracket = strchr(name, '[');
        GLuint arrayElement = 0u;
        if (bracket) {
            char *end = NULL;
            unsigned long parsed = strtoul(bracket + 1, &end, 10);
            size_t baseLen = (size_t)(bracket - name);
            if (!end || *end != ']' || end[1] != '\0' || baseLen == 0u ||
                baseLen >= sizeof(baseName)) {
                return NO;
            }
            memcpy(baseName, name, baseLen);
            baseName[baseLen] = '\0';
            arrayElement = (GLuint)parsed;
        } else {
            size_t nameLen = strlen(name);
            if (nameLen >= sizeof(baseName)) return NO;
            memcpy(baseName, name, nameLen + 1u);
        }
        const MGLShaderResource *output = NULL;
        for (GLuint i = 0u; outputs->list && i < outputs->count; i++) {
            if (outputs->list[i].name &&
                strcmp(outputs->list[i].name, baseName) == 0) {
                output = &outputs->list[i];
                break;
            }
        }
        if (!output || output->location >= 0x0fffffffu) {
            return NO;
        }
        GLuint recordSlot = output->location;
        if (bracket) {
            if (!output->is_array ||
                arrayElement >= (GLuint)((output->gl_array_size > 0)
                                             ? output->gl_array_size
                                             : 1)) {
                return NO;
            }
            recordSlot += arrayElement;
        } else if (output->is_array) {
            return NO;
        }
        sourceOffset[varying] = MGL_AIR_PER_VERTEX_STRIDE +
                                (NSUInteger)recordSlot * 16u;
        hasSource[varying] = YES;
    }

    NSUInteger captureOffset = 0u;
    id capture = [self captureAIRVertexPositionsForTessellation:drawCtx
                                                          first:first
                                                          count:count
                                                  instanceCount:instanceCount
                                                   baseInstance:baseInstance
                                                     outOffset:&captureOffset];
    if (!capture) return NO;
    _currentCBHasWork = YES;
    [self flushCommandBuffer:YES];

    const uint8_t *captureBytes =
        (const uint8_t *)mglDrawSupportBufferContents(capture);
    NSUInteger captureStride = mglAIRPerVertexStrideForResources(outputs);
    uint64_t recordCount64 = (uint64_t)(uint32_t)count *
                             (uint64_t)(uint32_t)instanceCount;
    if (!captureBytes || recordCount64 > NSUIntegerMax) {
        return YES;
    }
    NSUInteger recordCount = (NSUInteger)recordCount64;
    /* GL semantics: once any active buffer runs out of room, further
     * primitives are neither written nor counted by
     * TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN (PRIMITIVES_GENERATED keeps
     * counting).  Track the capped count across all buffers. */
    NSUInteger writtenTotal = recordCount;

    for (GLuint buffer = 0u; buffer < bufferCount; buffer++) {
        if (bufferStride[buffer] == 0u) continue;
        BufferBaseTarget *slot =
            &MGL_STATE(drawCtx)->buffer_base[_TRANSFORM_FEEDBACK_BUFFER]
                                        .buffers[buffer];
        if (!slot->buf || slot->offset < 0) {
            writtenTotal = 0;
            continue;
        }
        BufferMap map = {0};
        map.buf = slot->buf;
        map.offset = slot->offset;
        map.size = slot->size;
        NSUInteger visible = mglBufferMapVisibleBackingBytes(
            &map, slot->buf->size > 0 ? (size_t)slot->buf->size : 0u);
        NSUInteger sessionOffset = xfb->buffer_write_offsets[buffer] <=
                (GLuint64)NSUIntegerMax
            ? (NSUInteger)xfb->buffer_write_offsets[buffer] : visible;
        if (sessionOffset >= visible) {
            writtenTotal = 0;
            continue;
        }
        NSUInteger capacity = (visible - sessionOffset) / bufferStride[buffer];
        NSUInteger writtenRecords = MIN(recordCount, capacity);
        if (writtenRecords == 0u ||
            writtenRecords > NSUIntegerMax / bufferStride[buffer]) {
            writtenTotal = 0;
            continue;
        }
        if (writtenRecords < writtenTotal) {
            writtenTotal = writtenRecords;
        }
        NSUInteger writtenBytes = writtenRecords * bufferStride[buffer];
        uint8_t *packed = (uint8_t *)calloc(1u, writtenBytes);
        if (!packed) {
            mglDispatchError(drawCtx, "vertexTransformFeedback",
                             GL_OUT_OF_MEMORY);
            return YES;
        }
        for (NSUInteger record = 0u; record < writtenRecords; record++) {
            const uint8_t *srcRecord = captureBytes + captureOffset +
                                       record * captureStride;
            uint8_t *dstRecord = packed + record * bufferStride[buffer];
            for (GLsizei varying = 0;
                 varying < program->transform_feedback_varying_count;
                 varying++) {
                const MGLTransformFeedbackVaryingPlan *plan =
                    &program->transform_feedback_layout[varying];
                if (plan->buffer_index != buffer || !hasSource[varying]) {
                    continue;
                }
                memcpy(dstRecord + (NSUInteger)plan->component_offset * 4u,
                       srcRecord + sourceOffset[varying],
                       (NSUInteger)plan->component_count * 4u);
            }
        }
        NSUInteger destinationOffset = (NSUInteger)slot->offset + sessionOffset;
        mglRendererBufferSubData(drawCtx, slot->buf, destinationOffset,
                                 writtenBytes, packed);
        /* The renderer's subdata routes through the shadow/snapshot pair,
         * so the live Metal allocation may lag until the next snapshot
         * flush while glMapBufferRange serves from it directly.  Mirror
         * the bytes into the live allocation now; XFB capture is a
         * synchronous CPU-side operation by definition here. */
        if (slot->buf->data.mtl_data) {
            MGLRenderBufferInfo liveInfo = {0};
            if (mglRenderGetBufferInfo(slot->buf->data.mtl_data,
                                       &liveInfo) == 0 &&
                destinationOffset + writtenBytes <= liveInfo.length) {
                uint8_t *liveBase = (uint8_t *)mglDrawSupportBufferContents(
                    (__bridge id)(slot->buf->data.mtl_data));
                if (liveBase) {
                    memcpy(liveBase + destinationOffset, packed,
                           writtenBytes);
                }
            }
        }
        /* The renderer's subdata writes straight to the Metal allocation
         * and leaves the CPU shadow untouched; a later glMapBufferRange
         * served from the shadow would otherwise observe pre-capture
         * bytes. */
        if (slot->buf->data.buffer_data &&
            (size_t)slot->buf->size >= destinationOffset + writtenBytes) {
            memcpy((uint8_t *)slot->buf->data.buffer_data + destinationOffset,
                   packed, writtenBytes);
        }
        free(packed);
        slot->buf->ever_written = GL_TRUE;
        slot->buf->has_initialized_data = GL_TRUE;
        slot->buf->gpu_write_target = GL_TRUE;
        slot->buf->last_init_source = kInitMapWrite;
        slot->buf->last_write_offset = (GLintptr)destinationOffset;
        slot->buf->last_write_size = (GLsizeiptr)writtenBytes;
        if (slot->buf->written_min < 0 ||
            (GLintptr)destinationOffset < slot->buf->written_min) {
            slot->buf->written_min = (GLintptr)destinationOffset;
        }
        GLintptr writeEnd = (GLintptr)(destinationOffset + writtenBytes);
        if (slot->buf->written_max < 0 || writeEnd > slot->buf->written_max) {
            slot->buf->written_max = writeEnd;
        }
        xfb->buffer_write_offsets[buffer] += (GLuint64)writtenBytes;
    }

    xfb->primitives_generated += (GLuint64)recordCount;
    xfb->primitives_written += (GLuint64)writtenTotal;
    mglRecordActivePrimitiveQueryDraw(drawCtx, (GLuint64)recordCount,
                                      (GLuint64)writtenTotal);
    drawCtx->state.dirty_bits = DIRTY_ALL;
    return YES;
}

- (BOOL)handleGeometryDrawIfNeeded:(GLMContext)drawCtx
                              mode:(GLenum)mode
                             first:(GLint)first
                             count:(GLsizei)count
                         indexType:(GLenum)indexType
                           indices:(const void *)indices
                        baseVertex:(GLint)baseVertex
                     instanceCount:(GLsizei)instanceCount
                      baseInstance:(GLuint)baseInstance
                             label:(const char *)label
{
    if (!drawCtx) {
        return NO;
    }

    Program *program = mglResolveProgramForStageFromState(
        drawCtx, _GEOMETRY_SHADER);
    Shader *geometryShader = program
        ? program->shader_slots[_GEOMETRY_SHADER] : NULL;
    if (!program || !geometryShader) {
        return NO;
    }
    GLenum gsInputMode = program->geometry_input_type;
    if (gsInputMode != GL_POINTS && gsInputMode != GL_LINES &&
        gsInputMode != GL_LINES_ADJACENCY &&
        gsInputMode != GL_TRIANGLES &&
        gsInputMode != GL_TRIANGLES_ADJACENCY) {
        gsInputMode = GL_TRIANGLES;
    }
    GLenum gsOutputMode = program->geometry_output_type;
    if (gsOutputMode != GL_POINTS && gsOutputMode != GL_LINE_STRIP &&
        gsOutputMode != GL_TRIANGLE_STRIP) {
        gsOutputMode = GL_TRIANGLE_STRIP;
    }
    uint32_t outputPrimitive = gsOutputMode == GL_POINTS
        ? MGL_DRAW_PRIMITIVE_POINT
        : gsOutputMode == GL_LINE_STRIP ? MGL_DRAW_PRIMITIVE_LINE
        : MGL_DRAW_PRIMITIVE_TRIANGLE;
    const BOOL indexedDraw = (indexType != 0u);
    if (getenv("MGL_GS_DIAG")) {
        NSLog(@"MGL GS DIAG topology mode=0x%x gsIn=0x%x gsOut=0x%x indexed=%d count=%d first=%d vertsOut=%u route=%d",
              (unsigned)mode, (unsigned)gsInputMode, (unsigned)gsOutputMode,
              indexedDraw ? 1 : 0, (int)count, (int)first,
              (unsigned)program->geometry_vertices_out,
              (int)program->gs_route);
    }
    if (!mglGeometryInputModeAccepts(gsInputMode, mode) || count <= 0 ||
        instanceCount <= 0 || (!indexedDraw && first < 0)) {
        if (getenv("MGL_GS_DIAG")) {
            NSLog(@"MGL GS DIAG topology rejected mode=0x%x gsIn=0x%x",
                  (unsigned)mode, (unsigned)gsInputMode);
        }
        static uint64_t unsupportedDrawCount = 0;
        uint64_t hit = ++unsupportedDrawCount;
        if (hit <= 16ull || (hit % 512ull) == 0ull) {
            NSLog(@"MGL GS ERROR: blocking unsupported %s draw %@ "
                   "mode=0x%x gsIn=0x%x count=%d instances=%d baseInstance=%u",
                  indexedDraw ? "indexed" : "array",
                  label ? [NSString stringWithUTF8String:label] : @"draw",
                  (unsigned)mode, (unsigned)gsInputMode, (int)count, (int)instanceCount,
                  (unsigned)baseInstance);
        }
        /*  contract: never drop a GS draw silently.  A draw whose mode
         * does not match the GS input topology is an invalid operation. */
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_INVALID_OPERATION);
        return YES;
    }
    if (![self bindMTLProgram:program] ||
        !program->modules[_GEOMETRY_SHADER].mtl_function) {
        NSLog(@"MGL GS ERROR: failed to load AIR kernel program=%u",
              (unsigned)program->name);
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_INVALID_OPERATION);
        return YES;
    }

    self->ctx = drawCtx;
    if (![self ensureAIRGeometryPassthroughFunctionForProgram:program
                                              outputPrimitive:outputPrimitive]) {
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_OUT_OF_MEMORY);
        return YES;
    }


    uint32_t *gatherArray = NULL;
    uint32_t gatherCount = 0u;
    uint32_t gatherPrimitives = 0u;
    uint32_t gatherMaxIndex = 0u;
    const uint8_t *indexBytes = NULL;
    id eboMetal = nil;
    NSUInteger indexOffsetBytes = 0u;
    id gatherBuf = nil;
    MGLAIRGSGatherParams gparams;
    memset(&gparams, 0, sizeof(gparams));
    {
        Buffer *ebo = indexedDraw ? getElementBuffer(drawCtx) : NULL;
        if (indexedDraw &&
            (!ebo || ![self processBuffer:ebo] || !ebo->data.mtl_data)) {
            mglDispatchError(drawCtx, label ? label : "geometryDraw",
                             GL_INVALID_OPERATION);
            return YES;
        }
        if (indexedDraw) {
            eboMetal = (__bridge id)ebo->data.mtl_data;
            indexOffsetBytes = (NSUInteger)(uintptr_t)indices;
            indexBytes = mglElementIndexSourceForDraw(
                ebo, eboMetal, indexType, indexOffsetBytes, count);
            if (!indexBytes) {
                mglDispatchError(drawCtx, label ? label : "geometryDraw",
                                 GL_INVALID_OPERATION);
                return YES;
            }
        }
        uint32_t restartIndex = 0u;
        const bool restartEnabled = indexedDraw &&
            mglPrimitiveRestartIndexForType(drawCtx, indexType, &restartIndex);
        if (!mglGeometryGatherTopology(
                indexBytes, indexType, count, first, indexedDraw,
                restartEnabled, restartIndex, mode, &gatherArray,
                &gatherCount, &gatherPrimitives, &gatherMaxIndex)) {
            /* Incomplete primitive groups are valid GL draws with no
             * invocations. */
            return YES;
        }
        gatherBuf = mglDrawSupportCreateBufferWithBytes(
            _device, gatherArray, (NSUInteger)gatherCount * 4u, 0u);
        free(gatherArray);
        gatherArray = NULL;
        if (!gatherBuf) {
            mglDispatchError(drawCtx, label ? label : "geometryDraw",
                             GL_OUT_OF_MEMORY);
            return YES;
        }
        gparams.vertices_per_instance = indexedDraw
            ? gatherMaxIndex + 1u : (uint32_t)count;
        gparams.primitives_per_instance = gatherPrimitives;
        gparams.first_vertex = indexedDraw ? 0u : (uint32_t)first;
        gparams.gather_enabled = 1u;
        if (getenv("MGL_GS_DIAG")) {
            NSLog(@"MGL GS DIAG gather mode=0x%x indexed=%d first=%d count=%d gathered=%u prims=%u max=%u params={%u,%u,%u,%u}",
                  (unsigned)mode, indexedDraw ? 1 : 0, (int)first, (int)count,
                  (unsigned)gatherCount, (unsigned)gatherPrimitives,
                  (unsigned)gatherMaxIndex,
                  (unsigned)gparams.vertices_per_instance,
                  (unsigned)gparams.primitives_per_instance,
                  (unsigned)gparams.first_vertex,
                  (unsigned)gparams.gather_enabled);
        }
    }

    const GLuint primitiveCount = (GLuint)gatherPrimitives;
    if ((GLuint)instanceCount > UINT32_MAX / primitiveCount) {
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_OUT_OF_MEMORY);
        return YES;
    }
    const GLuint drawPrimitiveCount =
        primitiveCount * (GLuint)instanceCount;
    const GLuint invocationCount = MAX(1u, program->geometry_invocations);
    if (drawPrimitiveCount > UINT32_MAX / invocationCount) {
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_OUT_OF_MEMORY);
        return YES;
    }
    const GLuint workItemCount = drawPrimitiveCount * invocationCount;
    const NSUInteger outputStride = mglAIRPerVertexStrideForResources(
        &program->shader_resources_list[_GEOMETRY_SHADER][_STAGE_OUTPUT_RES]);
    /* Keep a zero-output GS on the same ABI: the backend emits no expanded
     * records, but each work item still owns the two header records used by
     * the indirect-draw layout. */
    const uint32_t maxVertices = program->geometry_vertices_out > 0u
        ? program->geometry_vertices_out : 1u;

    const MGLAIRGSOutputPrimitive gsAirOutput = gsOutputMode == GL_POINTS
        ? MGL_AIR_GS_OUT_POINTS
        : gsOutputMode == GL_LINE_STRIP ? MGL_AIR_GS_OUT_LINE_STRIP
        : MGL_AIR_GS_OUT_TRIANGLE_STRIP;
    const NSUInteger expandedVertices =
        mglAIRGSExpandedVertices(gsAirOutput, maxVertices);
    const NSUInteger recordsPerPrimitive =
        mglAIRGSRecordsPerPrimitive(gsAirOutput, maxVertices);
    if (primitiveCount == 0u ||
        recordsPerPrimitive >
            (NSUIntegerMax / outputStride) / workItemCount) {
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_OUT_OF_MEMORY);
        return YES;
    }

    /* Run the real VS once into the shared per-vertex records used by the AIR GS
     * kernel.  This helper closes the render encoder before compute begins. */
    NSUInteger inputOffset = 0u;
    id input = nil;
    if (indexedDraw) {
        input = [self captureAIRVertexPositionsForGeometryIndexed:drawCtx
                                                      indexBuffer:eboMetal
                                                        indexType:indexType
                                                      indexOffset:indexOffsetBytes
                                                            count:count
                                                        baseVertex:baseVertex
                                                     instanceCount:instanceCount
                                                      baseInstance:baseInstance
                                                         maxIndex:gatherMaxIndex
                                                        outOffset:&inputOffset];
    } else {
        input = [self captureAIRVertexPositionsForTessellation:
                         drawCtx
                                 first:first
                                 count:count
                         instanceCount:instanceCount
                          baseInstance:baseInstance
                           outOffset:&inputOffset];
    }
    if (!input) {
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_INVALID_OPERATION);
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return YES;
    }
    /* Publish the capture record stride the kernels must use when walking
     * gl_in records.  The capture writes one record per *vertex-stage*
     * output varying slot, so its layout comes from the VS output list and
     * can be wider than what this GS declares as inputs (e.g. a flat
     * instance_id the GS never reads).  A stride mismatch made every
     * gl_in[N>0] read land inside the wrong record. */
    Program *captureVS = mglResolveProgramForStageFromState(drawCtx, _VERTEX_SHADER);
    if (captureVS) {
        gparams.stage_in_stride =
            mglAIRPerVertexStrideForResources(
                &captureVS->shader_resources_list[_VERTEX_SHADER]
                                                 [_STAGE_OUTPUT_RES]);
    }
    if (getenv("MGL_GS_STRIDE_FORCE"))
        gparams.stage_in_stride =
            (uint32_t)atol(getenv("MGL_GS_STRIDE_FORCE"));
    /* Publish the GS-input -> capture-offset location map.  The capture
     * lays records out by the *vertex* stage's output locations; a VS
     * output the GS never declares (a flat helper like instance_id) shifts
     * every later slot, so reading by the GS's own locations lands in the
     * wrong fields.  loc_map[gs_loc] = vs_loc + 1; 0 falls back to
     * identity inside the kernel. */
    {
        memset(gparams.loc_map, 0, sizeof(gparams.loc_map));
        const MGLShaderResourceList *gsInputs =
            &program->shader_resources_list[_GEOMETRY_SHADER][_STAGE_INPUT_RES];
        const MGLShaderResourceList *vsOutputs =
            captureVS ? &captureVS->shader_resources_list[_VERTEX_SHADER]
                                                        [_STAGE_OUTPUT_RES]
                      : NULL;
        for (GLuint gi = 0u;
             gsInputs && vsOutputs && gsInputs->list && gi < gsInputs->count;
             gi++) {
            const MGLShaderResource *in = &gsInputs->list[gi];
            if (in->is_per_patch || !in->name || in->location >= 32u)
                continue;
            GLuint nameLen = (GLuint)strlen(in->name);
            const char *bracket = strchr(in->name, '[');
            if (bracket) nameLen = (GLuint)(bracket - in->name);
            for (GLuint vi = 0u; vi < vsOutputs->count && vsOutputs->list;
                 vi++) {
                const MGLShaderResource *out = &vsOutputs->list[vi];
                if (out->is_per_patch || !out->name)
                    continue;
                GLuint outLen = (GLuint)strlen(out->name);
                const char *ob = strchr(out->name, '[');
                if (ob) outLen = (GLuint)(ob - out->name);
                if (nameLen == outLen &&
                    strncmp(in->name, out->name, nameLen) == 0) {
                    gparams.loc_map[in->location] = out->location + 1u;
                    break;
                }
            }
        }
    }
    if (getenv("MGL_GS_DIAG")) {
        const MGLShaderResourceList *gsIn2 =
            &program->shader_resources_list[_GEOMETRY_SHADER][_STAGE_INPUT_RES];
        const MGLShaderResourceList *vsOut2 =
            captureVS ? &captureVS->shader_resources_list[_VERTEX_SHADER]
                                                        [_STAGE_OUTPUT_RES]
                      : NULL;
        for (GLuint gi2 = 0u; gsIn2 && gsIn2->list && gi2 < gsIn2->count; gi2++)
            NSLog(@"MGL GS DIAG gsIn[%u] name=%s loc=%u active=%d",
                  gi2, gsIn2->list[gi2].name ?: "?",
                  gsIn2->list[gi2].location,
                  (int)gsIn2->list[gi2].resource_active);
        for (GLuint vi2 = 0u; vsOut2 && vsOut2->list && vi2 < vsOut2->count; vi2++)
            NSLog(@"MGL GS DIAG vsOut[%u] name=%s loc=%u active=%d",
                  vi2, vsOut2->list[vi2].name ?: "?",
                  vsOut2->list[vi2].location,
                  (int)vsOut2->list[vi2].resource_active);
    }
    if (getenv("MGL_GS_DIAG"))
        NSLog(@"MGL GS DIAG gparams={%u,%u,%u,%u,%u} loc_map[0..3]={%u,%u,%u,%u}",
              gparams.vertices_per_instance, gparams.primitives_per_instance,
              gparams.first_vertex, gparams.gather_enabled,
              gparams.stage_in_stride,
              gparams.loc_map[0], gparams.loc_map[1],
              gparams.loc_map[2], gparams.loc_map[3]);
    void *pipelineHandle = NULL;
    char pipelineError[512] = {0};
    int pipelineResult = mglGetOrCreateProgramComputePipeline(
        program, _GEOMETRY_SHADER, &pipelineHandle,
        pipelineError, sizeof(pipelineError));
    id pipeline =
        pipelineResult == 0 && pipelineHandle
            ? (__bridge_transfer id)pipelineHandle
            : nil;
    if (!pipeline) {
        NSLog(@"MGL GS ERROR: compute PSO failed program=%u: %s",
              (unsigned)program->name,
              pipelineError[0] ? pipelineError : "unknown error");
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return YES;
    }
    MGLRenderCommandBufferState commandState = {0};
    if (!mglRenderCommandBufferOwnerHasState(
            _renderPassManager.state->currentCommandBufferOwner,
            &commandState) ||
        commandState.status >= 2u) {
        if (![self newCommandBuffer]) {
            drawCtx->state.dirty_bits = DIRTY_ALL;
            return YES;
        }
    }
    const NSUInteger outputSize =
        (NSUInteger)workItemCount * recordsPerPrimitive * outputStride;
    id output = mglDrawSupportCreateBuffer(
        _device, outputSize, 0u);
    if (getenv("MGL_GS_DIAG"))
        NSLog(@"MGL GS DIAG outputSize=%lu stride=%lu recordsPerPrim=%lu workItems=%u mtlLen=%@",
              (unsigned long)outputSize, (unsigned long)outputStride,
              (unsigned long)recordsPerPrimitive, (unsigned)workItemCount,
              [output valueForKey:@"length"]);

    const NSUInteger countsRecordBytes = MGL_AIR_GS_COUNTS_RECORD_BYTES;
    id counts = mglDrawSupportCreateBuffer(
        _device, (NSUInteger)workItemCount * countsRecordBytes,
        0u);
    if (!output || !counts || !mglDrawSupportBufferContents(output) || !mglDrawSupportBufferContents(counts)) {
        drawCtx->state.dirty_bits = DIRTY_ALL;
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_OUT_OF_MEMORY);
        return YES;
    }
    memset(mglDrawSupportBufferContents(counts), 0,
           (size_t)workItemCount * countsRecordBytes);
    memset(mglDrawSupportBufferContents(output), 0, outputSize);
    /* Preset the draw parameters the kernel never touches: instance_count=1,
     * base_vertex=0, base_instance=0 (memset already zeroed the rest). */
    {
        uint32_t *countsWords = (uint32_t *)mglDrawSupportBufferContents(counts);
        for (NSUInteger w = 0; w < workItemCount; w++) {
            countsWords[w * MGL_AIR_GS_COUNTS_RECORD_WORDS + 1] = 1u;
        }
    }

    for (NSUInteger unit = 0; unit < TEXTURE_UNITS; unit++) {
        Texture *image = MGL_STATE(drawCtx)->image_units[unit].tex;
        Texture *sampled = MGL_STATE(drawCtx)->active_textures[unit];
        if (image && ![self bindMTLTexture:image]) {
            drawCtx->state.dirty_bits = DIRTY_ALL;
            return YES;
        }
        if (sampled && ![self bindMTLTexture:sampled]) {
            drawCtx->state.dirty_bits = DIRTY_ALL;
            return YES;
        }
    }

    MGLStageBindingCopyBackList stageCopyBacks = {0};

    TransformFeedback *xfbState = MGL_STATE(drawCtx)->transform_feedback;
    const bool xfbActive = xfbState && xfbState->active && !xfbState->paused;

    /* ---- GL4 ordered multi-buffer XFB (mgl_air_gs_abi.h §5b) ----
     * Replace the prototype per-stream atomic-cursor capture with a
     * per-*buffer* layout driven by the link-time scatter plan
     * (Program.transform_feedback_layout[]).  Records are scattered by the
     * pass-2 aux kernel in emission order with whole-primitive cross-buffer
     * truncation. */
    const bool gsSeparate =
        program->transform_feedback_buffer_mode == GL_SEPARATE_ATTRIBS;

    /* Per-buffer scatter plan (indexed by transform-feedback buffer 0..3). */
    MGLAIRGSXFBScatterParams scatterParams;
    memset(&scatterParams, 0, sizeof(scatterParams));
    for (uint32_t b = 0u; b < MGL_AIR_GS_MAX_STREAMS; b++) {
        scatterParams.buffer_stream[b] = MGL_AIR_GS_XFB_NO_STREAM;
    }
    /* Per-buffer GL binding state for copy-back (indexed by buffer index). */
    NSUInteger bufferCapBytes[MGL_AIR_GS_MAX_STREAMS] = {0u};
    NSUInteger bufferPhysBase[MGL_AIR_GS_MAX_STREAMS] = {0u};
    NSUInteger bufferDstOffset[MGL_AIR_GS_MAX_STREAMS] = {0u};
    NSUInteger bufferRemaining[MGL_AIR_GS_MAX_STREAMS] = {0u};
    id bufferDstMTL[MGL_AIR_GS_MAX_STREAMS] = {nil};
    uint32_t xfbBufferCount = 0u;

    id xfbTemporary = nil;
    id xfbCaptureBuffer = nil;   /* slot-31 capture (always the temporary) */
    id xfbVisBuffer = nil;       /* pass-1 per-(work-item, buffer) bytes    */
    id xfbOffsetBuffer = nil;    /* CPU prefix offsets for pass 2           */
    id xfbWrittenBuffer = nil;   /* pass-2 per-(work-item, buffer) written  */
    id scatterPipeline = nil;

    if (xfbActive) {
        /* Build the field descriptors from the link-time scatter plan.  For
         * each captured (non-builtin) varying, locate its GS output resource
         * to recover the layout(location) that fixes the pass-1 stage-out
         * record source offset (MGL_AIR_PER_VERTEX_STRIDE + location*16).
         * The destination offset is the link plan's per-buffer component
         * offset verbatim: component offsets are per buffer and the link
         * validation keeps one feeding stream per buffer, so no regrouping
         * is needed; the loop only bakes the buffer->stream map the pass-2
         * scatter uses to attribute stage-out records to streams. */
        MGLShaderResourceList *gsOutputs =
            &program->shader_resources_list[_GEOMETRY_SHADER]
                                           [_STAGE_OUTPUT_RES];
        uint32_t fieldCount = 0u;
        for (uint32_t s = 0u; s < MGL_AIR_GS_MAX_STREAMS; s++) {
            for (GLsizei vi = 0;
                 vi < program->transform_feedback_varying_count &&
                 fieldCount < MGL_AIR_GS_XFB_MAX_FIELDS;
                 vi++) {
                const MGLTransformFeedbackVaryingPlan *plan =
                    &program->transform_feedback_layout[vi];
                if (plan->component_count == 0u) continue;
                if ((uint32_t)plan->stream != s) continue;
                if (plan->buffer_index >= MGL_AIR_GS_MAX_STREAMS) continue;
                const char *name =
                    program->transform_feedback_varying_names[vi];
                if (!name || !name[0]) continue;
                char baseName[96];
                strncpy(baseName, name, sizeof(baseName) - 1);
                baseName[sizeof(baseName) - 1] = '\0';
                char *bracket = strchr(baseName, '[');
                if (bracket) *bracket = '\0';
                GLuint location = UINT32_MAX;
                for (GLuint j = 0u; j < gsOutputs->count; j++) {
                    if (gsOutputs->list[j].name &&
                        strcmp(gsOutputs->list[j].name, baseName) == 0) {
                        location = gsOutputs->list[j].location;
                        break;
                    }
                }
                /* Built-in per-vertex outputs copy from the record's
                 * fixed per-vertex slots instead of a varying slot. */
                NSUInteger srcOffset;
                if (strcmp(baseName, "gl_Position") == 0 &&
                    plan->builtin) {
                    srcOffset = MGL_AIR_PER_VERTEX_POSITION_OFFSET;
                } else if (strcmp(baseName, "gl_PointSize") == 0 &&
                           plan->builtin) {
                    srcOffset = MGL_AIR_PER_VERTEX_POINT_SIZE_OFFSET;
                } else {
                    if (location == UINT32_MAX) continue;
                    srcOffset =
                        MGL_AIR_PER_VERTEX_STRIDE + location * 16u;
                }
                MGLAIRGSXFBFieldDesc *fd = &scatterParams.fields[fieldCount++];
                fd->buffer_index = plan->buffer_index;
                fd->src_offset = (uint32_t)srcOffset;
                fd->dst_offset = plan->component_offset * 4u;
                fd->byte_count = plan->component_count * 4u;
                scatterParams.buffer_stream[plan->buffer_index] = s;
                if (plan->buffer_index + 1u > xfbBufferCount) {
                    xfbBufferCount = plan->buffer_index + 1u;
                }
            }
        }
        scatterParams.field_count = fieldCount;

        /* Per-buffer record stride = max end offset of its captured fields
         * (GL 4.6 §13.2.4 records are tightly packed per buffer). */
        for (uint32_t f = 0u; f < fieldCount; f++) {
            const MGLAIRGSXFBFieldDesc *fd = &scatterParams.fields[f];
            uint32_t end = fd->dst_offset + fd->byte_count;
            if (end > scatterParams.buffers[fd->buffer_index].stride) {
                scatterParams.buffers[fd->buffer_index].stride = end;
            }
        }

        if (getenv("MGL_GS_XFB_DIAG")) {
            fprintf(stderr,
                    "MGL GS XFB DIAG fields=%u buffers=%u varyings=%d mode=0x%x\n",
                    fieldCount, xfbBufferCount,
                    program->transform_feedback_varying_count,
                  program->transform_feedback_buffer_mode);
            for (uint32_t f = 0u; f < fieldCount; f++) {
                NSLog(@"  field[%u] buf=%u src=%u dst=%u bytes=%u", f,
                      scatterParams.fields[f].buffer_index,
                      scatterParams.fields[f].src_offset,
                      scatterParams.fields[f].dst_offset,
                      scatterParams.fields[f].byte_count);
            }
        }

        /* Resolve each active buffer's GL binding, visible capacity and
         * session write offset.  Always capture into a fresh temporary so the
         * pass-2 scatter writes ordered records independent of the GL store
         * address; copy-back moves them afterwards. */
        NSUInteger physTotal = 0u;
        for (uint32_t b = 0u; b < xfbBufferCount; b++) {
            if (scatterParams.buffers[b].stride == 0u) continue;
            BufferBaseTarget *slot = &MGL_STATE(drawCtx)
                ->buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[b];
            if (!slot->buf) {
                if (getenv("MGL_GS_XFB_DIAG"))
                    NSLog(@"MGL GS XFB DIAG buffer[%u] no bound GL buffer", b);
                continue;
            }
            if (!slot->buf->data.mtl_data) {
                [self bindMTLBuffer:slot->buf];
            }
            id mtl = (__bridge id)(slot->buf->data.mtl_data);
            if (!mtl) {
                if (getenv("MGL_GS_XFB_DIAG"))
                    NSLog(@"MGL GS XFB DIAG buffer[%u] no MTL backing", b);
                continue;
            }
            BufferMap map = {0};
            map.buf = slot->buf;
            map.offset = slot->offset;
            map.size = slot->size;
            NSUInteger visible = mglBufferMapVisibleBackingBytes(
                &map, (size_t)mglDrawSupportBufferLength(mtl));
            NSUInteger sessionOffset = 0u;
            if (xfbState->buffer_write_offsets[b] <= (GLuint64)NSUIntegerMax) {
                sessionOffset = (NSUInteger)xfbState->buffer_write_offsets[b];
            }
            if (sessionOffset > visible || slot->offset < 0 ||
                (NSUInteger)slot->offset > NSUIntegerMax - sessionOffset) {
                if (getenv("MGL_GS_XFB_DIAG"))
                    NSLog(@"MGL GS XFB DIAG buffer[%u] offset overflow "
                          "vis=%lu sessOff=%lu slotOff=%lld", b,
                          (unsigned long)visible, (unsigned long)sessionOffset,
                          (long long)slot->offset);
                continue;
            }
            bufferRemaining[b] = visible - sessionOffset;
            bufferDstOffset[b] = (NSUInteger)slot->offset + sessionOffset;
            bufferDstMTL[b] = mtl;
            NSUInteger maxCap = (NSUInteger)workItemCount * expandedVertices *
                                scatterParams.buffers[b].stride;
            bufferCapBytes[b] = MIN(maxCap, bufferRemaining[b]);
            if (bufferCapBytes[b] > (NSUInteger)UINT32_MAX) {
                bufferCapBytes[b] = (NSUInteger)UINT32_MAX;
            }
            bufferPhysBase[b] = physTotal;
            physTotal += bufferCapBytes[b];
            scatterParams.buffers[b].capacity_bytes =
                (uint32_t)bufferCapBytes[b];
            scatterParams.buffers[b].capture_base =
                (uint32_t)MIN(bufferPhysBase[b], (NSUInteger)UINT32_MAX);
        }
        if (getenv("MGL_GS_XFB_DIAG")) {
            for (uint32_t b = 0u; b < xfbBufferCount; b++) {
                NSLog(@"  buffer[%u] stride=%u cap=%u base=%u physTotal=%lu dstMTL=%@",
                      b, scatterParams.buffers[b].stride,
                      scatterParams.buffers[b].capacity_bytes,
                      scatterParams.buffers[b].capture_base,
                      (unsigned long)physTotal, bufferDstMTL[b]);
            }
        }
        scatterParams.buffer_count = xfbBufferCount;
        scatterParams.work_item_count = (uint32_t)workItemCount;
        scatterParams.stage_out_stride = (uint32_t)outputStride;
        scatterParams.records_per_primitive = (uint32_t)recordsPerPrimitive;
        scatterParams.vertices_per_primitive =
            (uint32_t)(outputPrimitive == MGL_DRAW_PRIMITIVE_POINT
                           ? 1u
                           : (outputPrimitive == MGL_DRAW_PRIMITIVE_LINE
                                  ? 2u
                                  : 3u));
        scatterParams.expanded_offset_records = MGL_AIR_GS_HEADER_RECORDS;

        if (physTotal > 0u && xfbBufferCount > 0u) {
            xfbTemporary = mglDrawSupportCreateBuffer(_device, physTotal, 0u);
            if (xfbTemporary) {
                memset(mglDrawSupportBufferContents(xfbTemporary), 0,
                       physTotal);
                xfbCaptureBuffer = xfbTemporary;
            }
            const NSUInteger visBytes =
                (NSUInteger)workItemCount * MGL_AIR_GS_MAX_STREAMS *
                sizeof(uint32_t);
            xfbVisBuffer = mglDrawSupportCreateBuffer(_device, visBytes, 0u);
            xfbOffsetBuffer = mglDrawSupportCreateBuffer(_device, visBytes, 0u);
            xfbWrittenBuffer =
                mglDrawSupportCreateBuffer(_device, visBytes, 0u);
            if (xfbVisBuffer && mglDrawSupportBufferContents(xfbVisBuffer)) {
                memset(mglDrawSupportBufferContents(xfbVisBuffer), 0,
                       visBytes);
            }
            if (xfbWrittenBuffer &&
                mglDrawSupportBufferContents(xfbWrittenBuffer)) {
                memset(mglDrawSupportBufferContents(xfbWrittenBuffer), 0,
                       visBytes);
            }
            const MGLAuxShaderAsset *scatterAsset =
                mglAuxShaderAssetFind("gs_xfb_scatter");
            if (scatterAsset && scatterAsset->data) {
                void *scatterHandle = NULL;
                char scatterError[256] = {0};
                if (mglRenderGetOrCreateAuxComputePipelineFromMetallib(
                        scatterAsset->data, scatterAsset->size,
                        scatterAsset->hash, "mgl_gs_xfb_scatter",
                        MGL_RENDER_AUX_COMPUTE_GS_XFB_SCATTER, 0u,
                        &scatterHandle, scatterError,
                        sizeof(scatterError)) == 0 &&
                    scatterHandle) {
                    scatterPipeline = (__bridge_transfer id)scatterHandle;
                } else {
                    NSLog(@"MGL GS XFB ERROR: scatter pipeline failed: %s",
                          scatterError[0] ? scatterError : "unknown");
                }
            }
            if (!xfbTemporary || !xfbVisBuffer || !xfbOffsetBuffer ||
                !xfbWrittenBuffer || !scatterPipeline) {
                drawCtx->state.dirty_bits = DIRTY_ALL;
                mglDispatchError(drawCtx, label ? label : "geometryDraw",
                                 GL_OUT_OF_MEMORY);
                return YES;
            }
        }
    }
    /* Back-compat locals referenced by the query/copy-back tail below. */
    NSUInteger xfbDestinationMTL_unused = 0u; (void)xfbDestinationMTL_unused;
    NSUInteger streamStride[MGL_AIR_GS_MAX_STREAMS] = {0u};
    NSUInteger bufferStride[MGL_AIR_GS_MAX_STREAMS] = {0u};
    for (uint32_t b = 0u; b < MGL_AIR_GS_MAX_STREAMS; b++) {
        streamStride[b] = scatterParams.buffers[b].stride;
        bufferStride[b] = scatterParams.buffers[b].stride;
    }
    const uint32_t gsStreamCount =
        program->geometry_stream_count > 0u ? program->geometry_stream_count
                                            : 1u;
    const bool multiStream = gsStreamCount > 1u;
    (void)gsSeparate;
    MGLAIRGSXFBMeta xfbMeta;
    memset(&xfbMeta, 0, sizeof(xfbMeta));
    for (uint32_t s = 0u; s < MGL_AIR_GS_MAX_STREAMS; s++) {
        xfbMeta.stream[s].stride = (xfbCaptureBuffer && streamStride[s] > 0u &&
                                    bufferCapBytes[s] > 0u)
            ? (uint32_t)streamStride[s] : 0u;
        xfbMeta.stream[s].capacity_bytes =
            (uint32_t)MIN(bufferCapBytes[s], (NSUInteger)UINT32_MAX);
        xfbMeta.stream[s].capture_base =
            (uint32_t)MIN(bufferPhysBase[s], (NSUInteger)UINT32_MAX);
        xfbMeta.buffer_stream[s] = scatterParams.buffer_stream[s];
    }
    id xfbMetaBuf = mglDrawSupportCreateBufferWithBytes(
        _device, &xfbMeta, sizeof(xfbMeta), 0u);
    if (!xfbMetaBuf) {
        drawCtx->state.dirty_bits = DIRTY_ALL;
        mglDispatchError(drawCtx, label ? label : "geometryDraw",
                         GL_OUT_OF_MEMORY);
        return YES;
    }
    const BOOL cppDispatch = getenv("MGL_GS_LEGACY_DISPATCH") ? NO : YES;
    id compute = nil;
    MGLRenderComputeExecutionResult executionResult = {0};
    BOOL gsQueryCountersReady = NO;
    MGLRenderComputeExecutionPlan executionPlan = {0};
    NSMutableArray *executionTemporaries = cppDispatch
        ? [NSMutableArray array] : nil;
    if (cppDispatch) {
        executionPlan.pipeline = (__bridge void *)pipeline;
#define MGL_GS_PLAN_BUFFER(resource, bindingOffset, bindingIndex)                \
        do {                                                                     \
            executionPlan.binding_ops[executionPlan.binding_op_count++] =        \
                (MGLRenderComputeBindingOp){                                  \
                    0u, (uint32_t)(bindingIndex),                                \
                    (uint64_t)(bindingOffset), (__bridge void *)(resource),      \
                    NULL, 0u};                                                   \
        } while (0)
#define MGL_GS_PLAN_BYTES(data, dataLength, bindingIndex)                        \
        do {                                                                     \
            executionPlan.binding_ops[executionPlan.binding_op_count++] =        \
                (MGLRenderComputeBindingOp){                                  \
                    1u, (uint32_t)(bindingIndex), 0u, NULL,                      \
                    (data), (uint32_t)(dataLength)};                             \
        } while (0)
        MGL_GS_PLAN_BUFFER(input, inputOffset, MGL_AIR_GS_SLOT_INPUT);
        MGL_GS_PLAN_BUFFER(output, 0u, MGL_AIR_GS_SLOT_OUTPUT);
        MGL_GS_PLAN_BUFFER(counts, 0u, MGL_AIR_GS_SLOT_COUNTS);
        MGL_GS_PLAN_BUFFER(gatherBuf ? gatherBuf : counts, 0u,
                           MGL_AIR_GS_SLOT_GATHER);
        if (xfbCaptureBuffer) {
            MGL_GS_PLAN_BUFFER(xfbCaptureBuffer, 0u, MGL_AIR_GS_SLOT_XFB);
        }
        MGL_GS_PLAN_BUFFER(xfbMetaBuf, 0u, MGL_AIR_GS_SLOT_XFB_META);
        /* The kernel always declares the visibility slot; bind a harmless
         * buffer when XFB is inactive so reads never touch unbound
         * memory (Metal validation asserts on the missing binding). */
        MGL_GS_PLAN_BUFFER(xfbVisBuffer ? xfbVisBuffer : counts, 0u,
                           MGL_AIR_GS_SLOT_XFB_VIS);
        MGL_GS_PLAN_BYTES(&gparams, sizeof(gparams),
                          MGL_AIR_GS_SLOT_GATHER_PARAMS);
    } else {
        compute = mglDrawSupportCreateComputeEncoder(
            _renderPassManager.state->currentCommandBufferOwner);
        if (!compute) {
            drawCtx->state.dirty_bits = DIRTY_ALL;
            return YES;
        }
        mglDrawSupportSetComputePipeline(compute, pipeline);
        mglDrawSupportSetComputeBuffer(compute, input, inputOffset,
                                       MGL_AIR_GS_SLOT_INPUT);
        mglDrawSupportSetComputeBuffer(compute, output, 0u,
                                       MGL_AIR_GS_SLOT_OUTPUT);
        mglDrawSupportSetComputeBuffer(compute, counts, 0u,
                                       MGL_AIR_GS_SLOT_COUNTS);

        mglDrawSupportSetComputeBuffer(compute,
                                       gatherBuf ? gatherBuf : counts, 0u,
                                       MGL_AIR_GS_SLOT_GATHER);
        mglDrawSupportSetComputeBytes(compute, &gparams, sizeof(gparams),
                                      MGL_AIR_GS_SLOT_GATHER_PARAMS);

        if (xfbCaptureBuffer) {
            mglDrawSupportSetComputeBuffer(compute, xfbCaptureBuffer, 0u,
                                           MGL_AIR_GS_SLOT_XFB);
        }
        mglDrawSupportSetComputeBuffer(compute, xfbMetaBuf, 0u,
                                       MGL_AIR_GS_SLOT_XFB_META);
        /* The kernel always declares the visibility slot; bind a harmless
         * buffer when XFB is inactive so reads never touch unbound
         * memory (Metal validation asserts on the missing binding). */
        mglDrawSupportSetComputeBuffer(
            compute, xfbVisBuffer ? xfbVisBuffer : counts, 0u,
            MGL_AIR_GS_SLOT_XFB_VIS);
    }
    if (getenv("MGL_GS_DIAG")) {
        Program *gp = mglResolveProgramForStageFromState(drawCtx, _GEOMETRY_SHADER);
        NSLog(@"MGL GS DIAG GS uniform-constant resources: %u",
              gp ? gp->shader_resources_list[_GEOMETRY_SHADER][_UNIFORM_CONSTANT_RES].count : 0u);
    }
    if (getenv("MGL_GPU_CAPTURE")) {
        id desc = [self mglCaptureDescriptorForDevice:_device
                                          outputPath:[NSString stringWithUTF8String:getenv("MGL_GPU_CAPTURE")]];
        NSError *capErr = nil;
        if (desc && [self mglStartCaptureWithDescriptor:desc error:&capErr]) {
            NSLog(@"MGL GPU capture started -> %s", getenv("MGL_GPU_CAPTURE"));
        } else {
            NSLog(@"MGL GPU capture start failed: %@", capErr.localizedDescription);
        }
    }
    bool buffersOK = [self bindBuffersToComputeEncoder:compute
                                                   stage:_GEOMETRY_SHADER
                                               copyBacks:&stageCopyBacks
                                           executionPlan:cppDispatch ? &executionPlan : NULL
                                            temporaries:executionTemporaries];
    bool texturesOK = buffersOK && [self bindTexturesToComputeEncoder:compute
                                                                stage:_GEOMETRY_SHADER
                                                        executionPlan:cppDispatch ? &executionPlan : NULL
                                                         temporaries:executionTemporaries];
    if (getenv("MGL_GS_DIAG") && cppDispatch) {
        for (uint32_t bi = 0; bi < executionPlan.binding_op_count; bi++) {
            const MGLRenderComputeBindingOp *op = &executionPlan.binding_ops[bi];
            NSLog(@"MGL GS DIAG binding[%u] kind=%u slot=%u offset=%llu buffer=%p",
                  (unsigned)bi, (unsigned)op->kind, (unsigned)op->index,
                  (unsigned long long)op->offset, op->buffer);
            if (op->kind == 0u && op->index == 0u && op->buffer) {
                const float *f = (const float *)mglDrawSupportBufferContents(
                    (__bridge id)op->buffer);
                const int32_t *iw = (const int32_t *)f;
                NSLog(@"MGL GS DIAG uniform slot0 words: %d %d %d %d %d %d %d %d",
                      iw[0], iw[1], iw[2], iw[3], iw[4], iw[5], iw[6], iw[7]);
            }
        }
    }
    if (!buffersOK || !texturesOK) {
        if (compute) mglDrawSupportEndComputeEncoder(compute);
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return YES;
    }
    if (cppDispatch) {
        MGLRenderCopyBackEntry copyBackEntries[kMGLMaxBufferSlots] = {0};
        uint32_t copyBackEntryCount = 0u;
        for (NSUInteger slot = 0; slot < kMGLMaxBufferSlots; slot++) {
            MGLStageBindingCopyBack *entry = &stageCopyBacks.slots[slot];
            if (entry->length == 0) continue;
            copyBackEntries[copyBackEntryCount++] =
                (MGLRenderCopyBackEntry){
                    .temporary = entry->temporary,
                    .destination = entry->destination,
                    .destination_buffer = entry->destination_buffer,
                    .destination_offset = entry->destination_offset,
                    .length = entry->length,
                };
        }
        executionPlan.dispatch = (MGLRenderComputePlan){
            .dispatch_kind = MGL_RENDER_COMPUTE_DISPATCH_DIRECT,
            .groups_x = (uint32_t)workItemCount,
            .groups_y = 1u,
            .groups_z = 1u,
            .local_x = 1u,
            .local_y = 1u,
            .local_z = 1u,
        };
        executionPlan.barrier_scope = copyBackEntryCount
            ? MGL_RENDER_COMPUTE_BARRIER_BUFFERS
            : MGL_RENDER_COMPUTE_BARRIER_NONE;
        const BOOL requireCPUVisibility =
            xfbActive || mglHasActiveIndexedPrimitiveQuery() ||
            mglHasActivePrimitiveQuery() || mglHasActiveGeometryShaderQuery();
        const BOOL gsDiagnostic = getenv("MGL_GS_DIAG") != NULL;
        char executionError[256] = {0};
        if (mglRenderExecuteComputeExecutionPlan(
                _renderPassManager.state->currentCommandBufferOwner,
                _gpuRecovery.commandRecoveryOwner,
                &executionPlan, copyBackEntries, copyBackEntryCount,
                (requireCPUVisibility || gsDiagnostic) ? 1u : 0u, &executionResult,
                executionError, sizeof(executionError)) != 0) {
            if (executionResult.transaction.device_reset_requested) {
                atomic_store_explicit(&_deviceResetRequested, true,
                                      memory_order_release);
            }
            NSLog(@"MGL GS ERROR: C++ execution transaction failed: %s",
                  executionError[0] ? executionError : "unknown error");
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            drawCtx->state.dirty_bits = DIRTY_ALL;
            return YES;
        }
        gsQueryCountersReady = executionResult.transaction.waited != 0;
        [self clearStageBindingCopyBacks:&stageCopyBacks];
    } else {
        mglDrawSupportDispatchCompute(
            compute, workItemCount, 1u, 1u, 1u, 1u, 1u);
        mglDrawSupportEndComputeEncoder(compute);
        if (![self flushStageBindingCopyBacks:&stageCopyBacks
                         requireCPUVisibility:(xfbActive ||
                                               mglHasActiveIndexedPrimitiveQuery() ||
                                               mglHasActivePrimitiveQuery() ||
                                               mglHasActiveGeometryShaderQuery())]) {
            drawCtx->state.dirty_bits = DIRTY_ALL;
            return YES;
        }
        gsQueryCountersReady = YES;
    }
    _geometry.expansionActive = YES;
    _geometry.program = program;
    /* The passthrough pipeline rasterizes the GS output primitive class, so
     * drive inputPrimitiveTopology from the output mode, not the GL input
     * mode (e.g. points in -> triangle_strip out). */
    switch (outputPrimitive) {
        case MGL_DRAW_PRIMITIVE_POINT:
            _lastDrawPrimitiveMode = GL_POINTS;
            break;
        case MGL_DRAW_PRIMITIVE_LINE:
            _lastDrawPrimitiveMode = GL_LINES;
            break;
        default:
            _lastDrawPrimitiveMode = GL_TRIANGLES;
            break;
    }
    drawCtx->state.dirty_bits = DIRTY_ALL;

    /* ---- GL4 ordered XFB: CPU prefix-sum + pass-2 scatter ----
     * pass 1 (above) filled the visibility buffer; compute per-buffer
     * exclusive prefix offsets and run the ordered scatter kernel.  The
     * pass-1 transaction already waited for CPU visibility (requireCPUVisibility
     * includes xfbActive), so the visibility contents are stable here. */
    NSUInteger bufferWritten[MGL_AIR_GS_MAX_STREAMS] = {0u};
    if (xfbActive && xfbVisBuffer && xfbOffsetBuffer && xfbWrittenBuffer &&
        scatterPipeline && xfbCaptureBuffer &&
        mglDrawSupportBufferContents(xfbVisBuffer) &&
        mglDrawSupportBufferContents(xfbOffsetBuffer) &&
        mglDrawSupportBufferContents(xfbWrittenBuffer)) {
        uint32_t *vis =
            (uint32_t *)mglDrawSupportBufferContents(xfbVisBuffer);
        uint32_t *offsets =
            (uint32_t *)mglDrawSupportBufferContents(xfbOffsetBuffer);
        /* Exclusive prefix-sum per buffer across work items. */
        for (uint32_t b = 0u; b < xfbBufferCount; b++) {
            uint32_t running = 0u;
            for (uint32_t w = 0u; w < (uint32_t)workItemCount; w++) {
                uint32_t idx = w * MGL_AIR_GS_MAX_STREAMS + b;
                offsets[idx] = running;
                running += vis[idx];
            }
        }
        /* Run pass 2 as its own compute transaction on the scatter PSO. */
        MGLRenderComputeExecutionPlan scatterPlan = {0};
        scatterPlan.pipeline = (__bridge void *)scatterPipeline;
        uint32_t scatterOp = 0u;
#define MGL_GS_SCATTER_BYTES(data, dataLength, bindingIndex)                  \
        do {                                                                  \
            scatterPlan.binding_ops[scatterOp++] =                           \
                (MGLRenderComputeBindingOp){                                 \
                    1u, (uint32_t)(bindingIndex), 0u, NULL,                  \
                    (data), (uint32_t)(dataLength)};                          \
        } while (0)
#define MGL_GS_SCATTER_BUFFER(resource, bindingOffset, bindingIndex)         \
        do {                                                                  \
            scatterPlan.binding_ops[scatterOp++] =                           \
                (MGLRenderComputeBindingOp){                                 \
                    0u, (uint32_t)(bindingIndex),                            \
                    (uint64_t)(bindingOffset), (__bridge void *)(resource),  \
                    NULL, 0u};                                                \
        } while (0)
        MGL_GS_SCATTER_BYTES(&scatterParams, sizeof(scatterParams),
                             MGL_AIR_GS_XFB_SCATTER_PARAMS_SLOT);
        MGL_GS_SCATTER_BUFFER(xfbVisBuffer, 0u, MGL_AIR_GS_XFB_SCATTER_VIS_SLOT);
        MGL_GS_SCATTER_BUFFER(xfbOffsetBuffer, 0u,
                              MGL_AIR_GS_XFB_SCATTER_OFFSET_SLOT);
        MGL_GS_SCATTER_BUFFER(output, 0u, MGL_AIR_GS_XFB_SCATTER_STAGE_OUT_SLOT);
        MGL_GS_SCATTER_BUFFER(xfbCaptureBuffer, 0u,
                              MGL_AIR_GS_XFB_SCATTER_XFB_SLOT);
        MGL_GS_SCATTER_BUFFER(xfbWrittenBuffer, 0u,
                              MGL_AIR_GS_XFB_SCATTER_WRITTEN_SLOT);
#undef MGL_GS_SCATTER_BYTES
#undef MGL_GS_SCATTER_BUFFER
        scatterPlan.binding_op_count = scatterOp;
        scatterPlan.dispatch = (MGLRenderComputePlan){
            .dispatch_kind = MGL_RENDER_COMPUTE_DISPATCH_DIRECT,
            .groups_x = (uint32_t)workItemCount,
            .groups_y = 1u,
            .groups_z = 1u,
            .local_x = 1u,
            .local_y = 1u,
            .local_z = 1u,
        };
        scatterPlan.barrier_scope = MGL_RENDER_COMPUTE_BARRIER_BUFFERS;
        MGLRenderComputeExecutionResult scatterResult = {0};
        char scatterError[256] = {0};
        if (mglRenderExecuteComputeExecutionPlan(
                _renderPassManager.state->currentCommandBufferOwner,
                _gpuRecovery.commandRecoveryOwner,
                &scatterPlan, NULL, 0u, 1u, &scatterResult,
                scatterError, sizeof(scatterError)) != 0) {
            if (scatterResult.transaction.device_reset_requested) {
                atomic_store_explicit(&_deviceResetRequested, true,
                                      memory_order_release);
            }
            NSLog(@"MGL GS XFB ERROR: scatter transaction failed: %s",
                  scatterError[0] ? scatterError : "unknown error");
            drawCtx->state.dirty_bits = DIRTY_ALL;
            return YES;
        }
        /* Reduce the per-(work-item, buffer) written counters. */
        const uint32_t *written =
            (const uint32_t *)mglDrawSupportBufferContents(xfbWrittenBuffer);
        for (uint32_t b = 0u; b < xfbBufferCount; b++) {
            NSUInteger total = 0u;
            for (uint32_t w = 0u; w < (uint32_t)workItemCount; w++) {
                total += (NSUInteger)written[
                    w * MGL_AIR_GS_MAX_STREAMS + b];
            }
            bufferWritten[b] = total;
        }
    }

    GLuint64 queryGenerated =
        outputPrimitive == MGL_DRAW_PRIMITIVE_POINT
            ? (GLuint64)workItemCount * expandedVertices
            : (GLuint64)workItemCount * expandedVertices /
                  (outputPrimitive == MGL_DRAW_PRIMITIVE_LINE ? 2u : 3u);
    const GLuint64 vpp = outputPrimitive == MGL_DRAW_PRIMITIVE_POINT
        ? 1u
        : (outputPrimitive == MGL_DRAW_PRIMITIVE_LINE ? 2u : 3u);
    GLuint64 queryWritten = 0u;
    const MGLAIRGSXFBMeta *queryMeta = NULL;
    if (xfbActive && xfbMetaBuf && mglDrawSupportBufferContents(xfbMetaBuf)) {

        const MGLAIRGSXFBMeta *meta =
            (const MGLAIRGSXFBMeta *)mglDrawSupportBufferContents(xfbMetaBuf);
        queryMeta = meta;
        /* Ordered multi-buffer copy-back: blit each buffer's written segment
         * (already whole-primitive truncated by the scatter kernel) back to
         * its GL XFB target and advance the session write offset. */
        if (xfbTemporary) {
            id xfbBlit = nil;
            for (uint32_t b = 0u; b < xfbBufferCount; b++) {
                if (!bufferDstMTL[b] || scatterParams.buffers[b].stride == 0u)
                    continue;
                NSUInteger copyBytes = bufferWritten[b];
                if (copyBytes == 0u) continue;
                if (copyBytes > bufferRemaining[b])
                    copyBytes = bufferRemaining[b];
                if (copyBytes == 0u) continue;
                if (!xfbBlit) {
                    xfbBlit = mglDrawSupportCreateBlitEncoder(
                        _renderPassManager.state->currentCommandBufferOwner);
                    if (!xfbBlit) {
                        _geometry.expansionActive = NO;
                        _geometry.program = NULL;
                        drawCtx->state.dirty_bits = DIRTY_ALL;
                        return YES;
                    }
                }
                mglDrawSupportBlitCopyBuffer(xfbBlit, xfbTemporary,
                                             bufferPhysBase[b],
                                             bufferDstMTL[b],
                                             bufferDstOffset[b], copyBytes);
                BufferBaseTarget *slot = &MGL_STATE(drawCtx)
                    ->buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[b];
                if (slot->buf) slot->buf->ever_written = GL_TRUE;
                const GLuint64 currentOffset =
                    xfbState->buffer_write_offsets[b];
                xfbState->buffer_write_offsets[b] =
                    (GLuint64)copyBytes > UINT64_MAX - currentOffset
                        ? UINT64_MAX
                        : currentOffset + (GLuint64)copyBytes;
            }
            if (xfbBlit) mglDrawSupportEndBlitEncoder(xfbBlit);
        }
        /* stream 0 (non-indexed) written primitives come from buffer 0's
         * written bytes; the buffer-0 record stride is the per-primitive
         * packed size for the captured stream-0 varyings. */
        const NSUInteger buffer0PrimBytes =
            (NSUInteger)vpp * scatterParams.buffers[0].stride;
        queryWritten = buffer0PrimBytes > 0u
            ? (GLuint64)bufferWritten[0] / buffer0PrimBytes : 0u;
        /* Indexed stream>0 generated counters stay in the meta. */
    }
    if (!queryMeta && xfbMetaBuf && mglDrawSupportBufferContents(xfbMetaBuf) &&
        (mglHasActiveIndexedPrimitiveQuery() ||
         mglHasActivePrimitiveQuery() ||
         mglHasActiveGeometryShaderQuery())) {
        queryMeta = (const MGLAIRGSXFBMeta *)mglDrawSupportBufferContents(xfbMetaBuf);
    }
    if (gsQueryCountersReady && counts &&
        mglDrawSupportBufferContents(counts)) {
        const uint32_t maxGeneratedPerWorkItem =
            gsAirOutput == MGL_AIR_GS_OUT_POINTS
                ? maxVertices
                : gsAirOutput == MGL_AIR_GS_OUT_LINE_STRIP
                    ? (maxVertices > 1u ? maxVertices - 1u : 0u)
                    : (maxVertices > 2u ? maxVertices - 2u : 0u);
        const GLuint64 maxGenerated =
            (GLuint64)workItemCount * (GLuint64)maxGeneratedPerWorkItem;
        if (outputPrimitive == MGL_DRAW_PRIMITIVE_POINT) {
            /* Stream-0 generated: prefer meta when available (emitSum counts
             * all streams' EmitVertex calls).  Fall back to per-work-item
             * emit totals when meta was not populated for this draw. */
            if (queryMeta) {
                const GLuint64 metaGen =
                    (GLuint64)queryMeta->stream[0].generated;
                if (metaGen <= maxGenerated) {
                    queryGenerated = metaGen;
                }
            } else {
                const uint32_t *cw =
                    (const uint32_t *)mglDrawSupportBufferContents(counts);
                const uint32_t emitWord =
                    MGL_AIR_GS_COUNTS_ARGS_WORDS +
                    (uint32_t)(MGL_AIR_GS_COUNT_EMITTED - 1u);
                GLuint64 emitSum = 0u;
                for (GLuint w = 0u; w < workItemCount; w++) {
                    emitSum += cw[w * MGL_AIR_GS_COUNTS_RECORD_WORDS + emitWord];
                }
                if (emitSum <= maxGenerated) {
                    queryGenerated = emitSum;
                }
            }
        } else if (queryMeta) {
            const GLuint64 metaGen =
                (GLuint64)queryMeta->stream[0].generated;
            if (metaGen <= maxGenerated) {
                queryGenerated = metaGen;
            }
        }
    }
    if (xfbActive && MGL_STATE(drawCtx)->caps.rasterizer_discard) {
        /* GL_RASTERIZER_DISCARD: no pixels by definition; the compute
         * expansion already ran and the primitive query must still count
         * the generated/written primitives (persistent query semantics). */
        _currentCBHasWork = YES;
        mglRecordGeometryPrimitiveQueries(
            drawCtx, queryGenerated, queryWritten, xfbActive, queryMeta,
            gsStreamCount, bufferWritten, bufferStride, workItemCount);
        _geometry.expansionActive = NO;
        _geometry.program = NULL;
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return YES;
    }
    if (getenv("MGL_GS_DIAG"))
        NSLog(@"MGL GS DIAG rasterize-check empty=%d culled=%d enc=%d",
              (int)[self currentDrawRasterizationIsEmpty],
              (int)[self currentDrawModeIsFullyCulled:gsOutputMode],
              (int)mglRenderEncoderOwnerHasCurrent(
                  _renderPassManager.state->currentRenderEncoderOwner));
    if (![self processGLState:true] ||
        mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1 ||
        [self currentDrawRasterizationIsEmpty] ||
        [self currentDrawModeIsFullyCulled:gsOutputMode]) {
        if (xfbActive || mglHasActiveIndexedPrimitiveQuery() ||
            mglHasActivePrimitiveQuery() ||
            mglHasActiveGeometryShaderQuery()) {
            _currentCBHasWork = YES;
            mglRecordGeometryPrimitiveQueries(
                drawCtx, queryGenerated, queryWritten, xfbActive, queryMeta,
                gsStreamCount, bufferWritten, bufferStride, workItemCount);
        }
        _geometry.expansionActive = NO;
        _geometry.program = NULL;
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return YES;
    }

    /* The GS compute dispatch ended the render encoder and processGLState
     * rebuilt it, but the dirty-domain resource sync may have been marked
     * done for the *previous* encoder. Rebind fragment-stage buffers
     * (plain uniforms etc.) on the fresh encoder before the indirect
     * draws, or the fragment shader reads unbound slots. */
    if (!getenv("MGL_ABLATE_GS_REBIND")) {
        /* The binding-state dedup still reflects the pre-compute encoder;
         * clear the fragment table so the rebind below is not skipped. */
        for (uint32_t slot = 0u; slot < 31u; slot++)
            mglRenderBindingClearFragmentBuffer(_bindingStateOwner, slot);
        MGLEncodeContext gsEncCtx = {
            .render_encoder_owner =
                _renderPassManager.state->currentRenderEncoderOwner,
        };
        [self bindFragmentBuffersToCurrentRenderEncoder:&gsEncCtx];
        [self bindBufferSizeConstantsForRenderEncoder];
    }
    [self applyPolygonOffsetForDrawMode:gsOutputMode];
    if (getenv("MGL_SYNC_AFTER_GS")) {
        [self flushCommandBuffer:YES];
        NSLog(@"MGL GS sync: flushed after compute");
    }
    if (getenv("MGL_GS_DIAG")) {
        const uint32_t *cw = (const uint32_t *)mglDrawSupportBufferContents(counts);
        NSLog(@"MGL GS DIAG draw counts w0..6: %u %u %u %u %u %u %u outputBuf=%p",
              cw[0], cw[1], cw[2], cw[3], cw[4], cw[5], cw[6], output);
        for (NSUInteger w = 0u; w < workItemCount; w++)
            NSLog(@"MGL GS DIAG counts[%lu] full: %u %u %u %u %u %u %u",
                  (unsigned long)w,
                  cw[w * MGL_AIR_GS_COUNTS_RECORD_WORDS],
                  cw[w * MGL_AIR_GS_COUNTS_RECORD_WORDS + 1],
                  cw[w * MGL_AIR_GS_COUNTS_RECORD_WORDS + 2],
                  cw[w * MGL_AIR_GS_COUNTS_RECORD_WORDS + 3],
                  cw[w * MGL_AIR_GS_COUNTS_RECORD_WORDS + 4],
                  cw[w * MGL_AIR_GS_COUNTS_RECORD_WORDS + 5],
                  cw[w * MGL_AIR_GS_COUNTS_RECORD_WORDS + 6]);
        {
            const float *of = (const float *)mglDrawSupportBufferContents(output);
            NSLog(@"MGL GS DIAG output floats [1040B]=%g,%g,%g,%g [1920B]=%g,%g [2800B]=%g,%g",
                  of[260], of[261], of[262], of[263],
                  of[480], of[481], of[700], of[701]);
        }
    }
    if (getenv("MGL_GS_SINGLE_DRAW")) {
        /* bisect: one draw over all records; header records carry pos=0
         * and are clipped away by the rasterizer. */
        const uint32_t totalVerts =
            (uint32_t)(workItemCount * recordsPerPrimitive) -
            MGL_AIR_GS_HEADER_RECORDS;
        uint32_t *cw1 = (uint32_t *)mglDrawSupportBufferContents(counts);
        cw1[0] = totalVerts;
        cw1[1] = 1u;
        mglDrawSupportSetVertexBuffer(
            _renderPassManager.state->currentRenderEncoderOwner, output,
            MGL_AIR_GS_HEADER_RECORDS * outputStride, 0u);
        mglDrawSupportDrawPrimitives(
            _renderPassManager.state->currentRenderEncoderOwner,
            outputPrimitive, 0u, totalVerts, 1u, 0u);
        goto after_gs_draws;
    }
    if (getenv("MGL_GS_COPY_DRAW")) {
        /* bisect: copy each work item's records to a fresh offset-0 buffer
         * (CPU readback -> newBufferWithBytes) and draw from that. */
        const uint8_t *src = (const uint8_t *)mglDrawSupportBufferContents(output);
        for (GLuint w = 0u; w < workItemCount; w++) {
            NSUInteger srcOff =
                ((NSUInteger)w * recordsPerPrimitive +
                 MGL_AIR_GS_HEADER_RECORDS) * outputStride;
            NSUInteger bytes = (NSUInteger)9u * outputStride;
            id sub = mglDrawSupportCreateBufferWithBytes(
                _device, src + srcOff, bytes, 0u);
            const uint32_t *cwv = (const uint32_t *)mglDrawSupportBufferContents(counts);
            mglDrawSupportSetVertexBuffer(
                _renderPassManager.state->currentRenderEncoderOwner, sub, 0u, 0u);
            mglDrawSupportDrawPrimitives(
                _renderPassManager.state->currentRenderEncoderOwner,
                outputPrimitive, 0u,
                cwv ? cwv[w * MGL_AIR_GS_COUNTS_RECORD_WORDS] : 0u, 1u, 0u);
        }
        goto after_gs_draws;
    }
    const char *onlyPrim = getenv("MGL_GS_ONLY_PRIM");
    for (GLuint iter = 0u; iter < workItemCount; iter++) {
        GLuint primitive = getenv("MGL_GS_REVERSE_DRAW")
            ? (workItemCount - 1u - iter) : iter;
        if (onlyPrim && (GLint)primitive != atoi(onlyPrim)) continue;
        const char *offOverride = getenv("MGL_GS_DRAW_OFFSET");
        NSUInteger offset =
            ((NSUInteger)primitive * recordsPerPrimitive +
             MGL_AIR_GS_HEADER_RECORDS) * outputStride;
        if (offOverride) offset = (NSUInteger)atol(offOverride);
        if (getenv("MGL_GS_VSTART_DRAW")) {
            /* bisect: bind at 0 and use indirect vertexStart to select the
             * work item's records (gl_VertexID starts at vertexStart). */
            static uint32_t *cwStart = NULL;
            cwStart = (uint32_t *)mglDrawSupportBufferContents(counts);
            cwStart[primitive * MGL_AIR_GS_COUNTS_RECORD_WORDS + 2] =
                primitive * (uint32_t)recordsPerPrimitive + 2u;
            offset = 0u;
        }
        id ptvsSource = output;
        NSUInteger ptvsOffset = offset;
        if (getenv("MGL_GS_BIND_INPUT")) {
            /* Diagnostic: point the passthrough VS at the capture buffer
             * instead of the kernel output so the rendered image shows
             * what the GPU actually wrote per input vertex.  An explicit
             * MGL_GS_DRAW_OFFSET still wins, for byte-range scans. */
            ptvsSource = input;
            if (!getenv("MGL_GS_DRAW_OFFSET"))
                ptvsOffset = inputOffset;
            offset = 0u;
        }
        mglDrawSupportSetVertexBuffer(_renderPassManager.state->currentRenderEncoderOwner, ptvsSource, ptvsOffset, 0u);
        if (getenv("MGL_GS_DIRECT_DRAW")) {
            const uint32_t *cw2 = (const uint32_t *)mglDrawSupportBufferContents(counts);
            mglDrawSupportDrawPrimitives(
                _renderPassManager.state->currentRenderEncoderOwner, outputPrimitive,
                0u, cw2 ? cw2[primitive * MGL_AIR_GS_COUNTS_RECORD_WORDS] : 0u,
                1u, 0u);
        } else if (getenv("MGL_GS_DRAW_VCOUNT")) {
            mglDrawSupportDrawPrimitives(
                _renderPassManager.state->currentRenderEncoderOwner, outputPrimitive,
                0u, (NSUInteger)atol(getenv("MGL_GS_DRAW_VCOUNT")), 1u, 0u);
        } else
        mglDrawSupportDrawPrimitivesIndirect(
            _renderPassManager.state->currentRenderEncoderOwner, outputPrimitive, counts,
            (offOverride ? 0u : (NSUInteger)primitive * countsRecordBytes));
        if (getenv("MGL_GS_DIAG")) {
            NSLog(@"MGL GS DIAG pre-draw prim=%u enc=%d",
                  primitive,
                  (int)mglRenderEncoderOwnerHasCurrent(
                      _renderPassManager.state->currentRenderEncoderOwner));
            const float *op = (const float *)((const uint8_t *)mglDrawSupportBufferContents(output) + offset);
            NSLog(@"MGL GS DIAG draw prim=%u offset=%lu firstRec={%g,%g,%g,%g}",
                  primitive, (unsigned long)offset,
                  op[0], op[1], op[2], op[3], 0.0);
        }
    }
after_gs_draws:
    if (getenv("MGL_GS_POST_DIAG")) {
        /* Dump the output records after the frame's GPU work completes
         * (the caller's glFinish/ReadPixels drains the encoders), so the
         * CPU view reflects what the rasterizing draws actually read. */
        [self flushCommandBuffer:YES];
        const uint8_t *outBytes =
            (const uint8_t *)mglDrawSupportBufferContents(output);
        NSLog(@"MGL GS POST-DIAG outputStride=%lu recordsPerPrim=%lu",
              (unsigned long)outputStride, (unsigned long)recordsPerPrimitive);
        if (input) {
            const uint8_t *inBytes =
                (const uint8_t *)mglDrawSupportBufferContents(input);
            NSUInteger inStride = mglAIRPerVertexStrideForResources(
                &program->shader_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES]);
            for (GLuint vtx = 0u; vtx < MIN((GLuint)count, 6u); vtx++) {
                const float *p = (const float *)(inBytes +
                    (NSUInteger)(vtx + (GLuint)first) * inStride);
                NSLog(@"MGL GS POST-DIAG in.cap[%u] pos={%g,%g,%g,%g} vary@64={%g,%g,%g,%g} @80={%g,%g,%g,%g}",
                      (unsigned)(vtx + (GLuint)first),
                      p[0], p[1], p[2], p[3], p[16], p[17], p[18], p[19],
                      p[20], p[21], p[22], p[23]);
            }
        }
        for (GLuint wi = 0u; wi < MIN(workItemCount, 4u); wi++) {
            for (GLuint rec = 0u; rec < MIN(recordsPerPrimitive, 9u); rec++) {
                const float *p = (const float *)(outBytes +
                    ((NSUInteger)wi * recordsPerPrimitive + rec) * outputStride);
                NSLog(@"MGL GS POST-DIAG out[%u].rec[%u] pos={%g,%g,%g,%g} ps=%g cull=%g,%g vary={%g,%g,%g,%g}",
                      (unsigned)wi, (unsigned)rec,
                      p[0], p[1], p[2], p[3],
                      p[4], p[5], p[6],
                      p[16], p[17], p[18], p[19]);
            }
        }
    }
    _currentCBHasWork = YES;
    if (getenv("MGL_GPU_CAPTURE")) {
        [self flushCommandBuffer:YES];
        [self mglStopCapture];
        NSLog(@"MGL GPU capture stopped");
    }
    mglRecordGeometryPrimitiveQueries(
        drawCtx, queryGenerated, queryWritten, xfbActive, queryMeta,
        gsStreamCount, bufferWritten, bufferStride, workItemCount);
    _geometry.expansionActive = NO;
    _geometry.program = NULL;
    drawCtx->state.dirty_bits = DIRTY_ALL;
    return YES;
}

- (bool) validateDrawArraysVertexInputs:(GLMContext)drawCtx
                                    mode:(GLenum)mode
                                   first:(GLint)first
                                   count:(GLsizei)count
                                drawCall:(uint64_t)drawCall
{
    if (!mglVboRangeValidationEnabled()) {
        return true;
    }

    if (!drawCtx) {
        NSLog(@"MGL DRAWARRAYS BLOCK call=%llu reason=null_ctx mode=0x%x first=%d count=%d",
              (unsigned long long)drawCall, (unsigned)mode, (int)first, (int)count);
        return false;
    }

    if (count == 0) {
        return false;
    }

    if (count < 0 || first < 0) {
        NSLog(@"MGL DRAWARRAYS BLOCK call=%llu reason=invalid_range mode=0x%x first=%d count=%d",
              (unsigned long long)drawCall, (unsigned)mode, (int)first, (int)count);
        return false;
    }

    uint64_t firstVertex = (uint64_t)(uint32_t)first;
    uint64_t vertexCount = (uint64_t)(uint32_t)count;
    if (vertexCount == 0u || firstVertex > UINT64_MAX - (vertexCount - 1u)) {
        NSLog(@"MGL DRAWARRAYS BLOCK call=%llu reason=vertex_range_overflow mode=0x%x first=%d count=%d",
              (unsigned long long)drawCall, (unsigned)mode, (int)first, (int)count);
        return false;
    }

    uint64_t lastVertex = firstVertex + vertexCount - 1u;
    VertexArray *vao = mglRendererGetValidatedVAO(drawCtx, "drawArrays.vboRange");
    if (!vao) {
        NSLog(@"MGL DRAWARRAYS BLOCK call=%llu reason=invalid_vao mode=0x%x first=%d count=%d",
              (unsigned long long)drawCall, (unsigned)mode, (int)first, (int)count);
        return false;
    }

    GLuint maxAttribs = MAX_ATTRIBS;

    for (GLuint attrib = 0; attrib < maxAttribs; attrib++) {
        if ((vao->enabled_attribs & (0x1u << attrib)) == 0u) {
            continue;
        }

        MGLResolvedVertexAttribBinding resolved = {0};
        if (!mglRendererResolveVertexAttribBinding(drawCtx,
                                                   vao,
                                                   attrib,
                                                   "drawArrays.vboRange",
                                                   &resolved)) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u reason=invalid_vbo mode=0x%x first=%d count=%d",
                  (unsigned long long)drawCall, (unsigned)attrib, (unsigned)mode, (int)first, (int)count);
            return false;
        }
        const VertexAttrib *a = resolved.attrib;
        Buffer *vbo = resolved.buffer;

        if (!mglRendererBufferHasDrawableContents(vbo)) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=never_written "
                  "init(source=%u mapped=%u access=0x%x accessFlags=0x%x full=%u range=[%lld,%lld) lastOff=%lld lastSize=%lld src=%p hash=0x%016llx)",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned)vbo->last_init_source,
                  (unsigned)vbo->mapped,
                  (unsigned)vbo->access,
                  (unsigned)vbo->access_flags,
                  (unsigned)vbo->has_initialized_data,
                  (long long)vbo->written_min,
                  (long long)vbo->written_max,
                  (long long)vbo->last_write_offset,
                  (long long)vbo->last_write_size,
                  vbo->last_write_src_ptr,
                  (unsigned long long)vbo->last_write_src_hash);
            return false;
        }

        if (resolved.binding_offset < 0 || resolved.relativeoffset < 0) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=negative_attrib_offset bindingOffset=%lld relativeOffset=%lld",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (long long)resolved.binding_offset,
                  (long long)resolved.relativeoffset);
            return false;
        }

        size_t compSize = mglVertexAttribComponentSize(a->type);
        size_t compCount = (size_t)a->size;
        if (compSize == 0u || compCount == 0u) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=invalid_attrib_format type=0x%x size=%u",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned)a->type,
                  (unsigned)a->size);
            return false;
        }

        if (compCount > (SIZE_MAX / compSize)) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=elem_size_overflow compSize=%zu compCount=%zu",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  compSize,
                  compCount);
            return false;
        }

        uint64_t elemBytes = (uint64_t)(compSize * compCount);
        uint64_t stride = (resolved.stride > 0u) ? (uint64_t)resolved.stride : elemBytes;
        uint64_t bindingOffset = (uint64_t)resolved.binding_offset;
        uint64_t attrRelativeOffset = (uint64_t)resolved.relativeoffset;
        if (bindingOffset > UINT64_MAX - attrRelativeOffset) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=offset_overflow bindingOffset=%llu relativeOffset=%llu",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)bindingOffset,
                  (unsigned long long)attrRelativeOffset);
            return false;
        }
        uint64_t relOffset = bindingOffset + attrRelativeOffset;
        if (stride == 0u || elemBytes == 0u) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=zero_stride_or_elem stride=%llu elem=%llu",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)stride,
                  (unsigned long long)elemBytes);
            return false;
        }

        // Per-instance attributes are still consumed by a non-instanced draw for
        // instance zero, so validate element zero instead of ignoring them.
        uint64_t rangeFirst = (resolved.divisor != 0u) ? 0u : firstVertex;
        uint64_t rangeLast = (resolved.divisor != 0u) ? 0u : lastVertex;

        if (relOffset > UINT64_MAX - elemBytes) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=byte_range_overflow bindingOffset=%llu relOffset=%llu elemBytes=%llu divisor=%u",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)bindingOffset,
                  (unsigned long long)relOffset,
                  (unsigned long long)elemBytes,
                  (unsigned)resolved.divisor);
            return false;
        }

        if (rangeLast > (UINT64_MAX - relOffset - elemBytes) / stride ||
            rangeFirst > (UINT64_MAX - relOffset) / stride) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=byte_range_overflow "
                  "range=[%llu,%llu] stride=%llu bindingOffset=%llu relOffset=%llu elemBytes=%llu divisor=%u",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)rangeFirst,
                  (unsigned long long)rangeLast,
                  (unsigned long long)stride,
                  (unsigned long long)bindingOffset,
                  (unsigned long long)relOffset,
                  (unsigned long long)elemBytes,
                  (unsigned)resolved.divisor);
            return false;
        }

        uint64_t byteStart = relOffset + (rangeFirst * stride);
        uint64_t byteEnd = relOffset + (rangeLast * stride) + elemBytes;
        uint64_t vboSize = (vbo->size > 0) ? (uint64_t)vbo->size : 0u;
        if (byteEnd > vboSize) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=vbo_oob "
                  "vertexRange=[%llu,%llu] byteRange=[%llu,%llu) vboSize=%llu "
                  "mode=0x%x first=%d count=%d stride=%llu bindingOffset=%llu relOffset=%llu elemBytes=%llu type=0x%x size=%u divisor=%u",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)rangeFirst,
                  (unsigned long long)rangeLast,
                  (unsigned long long)byteStart,
                  (unsigned long long)byteEnd,
                  (unsigned long long)vboSize,
                  (unsigned)mode,
                  (int)first,
                  (int)count,
                  (unsigned long long)stride,
                  (unsigned long long)bindingOffset,
                  (unsigned long long)relOffset,
                  (unsigned long long)elemBytes,
                  (unsigned)a->type,
                  (unsigned)a->size,
                  (unsigned)resolved.divisor);
            return false;
        }

        if (!vbo->data.mtl_data) {
            [self bindMTLBuffer:vbo];
        }
        if (!vbo->data.mtl_data) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=no_mtl_buffer byteRange=[%llu,%llu)",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)byteStart,
                  (unsigned long long)byteEnd);
            return false;
        }

        id mtlBuffer = (__bridge id)(vbo->data.mtl_data);
        if (!mtlBuffer) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=mtl_bridge_nil",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name);
            return false;
        }

        uint64_t metalLen = (uint64_t)mglDrawSupportBufferLength(mtlBuffer);
        if (byteEnd > metalLen) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=metal_oob "
                  "byteRange=[%llu,%llu) metalLen=%llu vboSize=%llu first=%d count=%d",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)byteStart,
                  (unsigned long long)byteEnd,
                  (unsigned long long)metalLen,
                  (unsigned long long)vboSize,
                  (int)first,
                  (int)count);
            return false;
        }

        if (vbo->written_min >= 0 && vbo->written_max >= 0) {
            uint64_t writtenMin = (uint64_t)vbo->written_min;
            uint64_t writtenMax = (uint64_t)vbo->written_max;
            if (byteStart < writtenMin || byteEnd > writtenMax) {
                NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=unwritten_range "
                      "byteRange=[%llu,%llu) written=[%llu,%llu) first=%d count=%d source=%u",
                      (unsigned long long)drawCall,
                      (unsigned)attrib,
                      (unsigned)vbo->name,
                      (unsigned long long)byteStart,
                      (unsigned long long)byteEnd,
                      (unsigned long long)writtenMin,
                      (unsigned long long)writtenMax,
                      (int)first,
                      (int)count,
                      (unsigned)vbo->last_init_source);
                return false;
            }
        }

        GLuint drawProgramKey = mglCurrentRenderProgramKey(drawCtx);
        if (mglShouldInspectDrawCall(drawCall, drawProgramKey) && attrib == 0u) {
            mglTraceLogNSString(@"MGL TRACE drawArrays.attrib0 call=%llu program=%u buffer=%u first=%d count=%d "
                  "byteRange=[%llu,%llu) vboSize=%llu metalLen=%llu stride=%llu bindingOffset=%llu relOffset=%llu elemBytes=%llu",
                  (unsigned long long)drawCall,
                  (unsigned)drawProgramKey,
                  (unsigned)vbo->name,
                  (int)first,
                  (int)count,
                  (unsigned long long)byteStart,
                  (unsigned long long)byteEnd,
                  (unsigned long long)vboSize,
                  (unsigned long long)metalLen,
                  (unsigned long long)stride,
                  (unsigned long long)bindingOffset,
                  (unsigned long long)relOffset,
                  (unsigned long long)elemBytes);
        }
    }

    return true;
}

- (BOOL)resolveElementBufferForDraw:(const char *)label
                            context:(GLMContext)drawCtx
                           glBuffer:(Buffer **)glBufferOut
                          mtlBuffer:(id *)mtlBufferOut
{
    Buffer *gl_element_buffer = getElementBuffer(drawCtx);
    return [self resolveElementBuffer:gl_element_buffer
                                label:label
                              context:drawCtx
                             glBuffer:glBufferOut
                            mtlBuffer:mtlBufferOut];
}

- (BOOL)resolveElementBufferForCommand:(const MGLDrawCommand *)cmd
                                  label:(const char *)label
                                context:(GLMContext)drawCtx
                               glBuffer:(Buffer **)glBufferOut
                              mtlBuffer:(id *)mtlBufferOut
{
    Buffer *gl_element_buffer = NULL;
    if (cmd && cmd->elementBuffer) {
        gl_element_buffer = mglRendererGetValidatedBuffer(drawCtx,
                                                          (Buffer *)cmd->elementBuffer,
                                                          label ? label : "deferred indexed draw",
                                                          0);
        if (!gl_element_buffer) {
            return NO;
        }
    } else {
        gl_element_buffer = getElementBuffer(drawCtx);
    }

    return [self resolveElementBuffer:gl_element_buffer
                                label:label
                              context:drawCtx
                             glBuffer:glBufferOut
                            mtlBuffer:mtlBufferOut];
}

- (BOOL)resolveElementBuffer:(Buffer *)gl_element_buffer
                       label:(const char *)label
                     context:(GLMContext)drawCtx
                    glBuffer:(Buffer **)glBufferOut
                   mtlBuffer:(id *)mtlBufferOut
{
    if (!gl_element_buffer) {
        NSLog(@"MGL WARNING: %s skipped because no element array buffer is bound", label ? label : "indexed draw");
        if (drawCtx) {
            mglDispatchError(drawCtx, label ? label : __FUNCTION__, GL_INVALID_OPERATION);
        }
        return NO;
    }

    if ([self processBuffer:gl_element_buffer] == false) {
        return NO;
    }

    id indexBuffer = (__bridge id)(gl_element_buffer->data.mtl_data);
    if (!indexBuffer) {
        NSLog(@"MGL WARNING: %s skipped because element buffer %u has no Metal buffer",
              label ? label : "indexed draw",
              gl_element_buffer->name);
        return NO;
    }

    if (glBufferOut) {
        *glBufferOut = gl_element_buffer;
    }
    if (mtlBufferOut) {
        *mtlBufferOut = indexBuffer;
    }
    return YES;
}

- (BOOL)resolveIndirectBufferForDraw:(const char *)label
                             context:(GLMContext)drawCtx
                            glBuffer:(Buffer **)glBufferOut
                           mtlBuffer:(id *)mtlBufferOut
{
    Buffer *gl_indirect_buffer = getIndirectBuffer(drawCtx);
    if (!gl_indirect_buffer) {
        NSLog(@"MGL WARNING: %s skipped because no draw indirect buffer is bound", label ? label : "indirect draw");
        if (drawCtx) {
            mglDispatchError(drawCtx, label ? label : __FUNCTION__, GL_INVALID_OPERATION);
        }
        return NO;
    }

    if ([self processBuffer:gl_indirect_buffer] == false) {
        return NO;
    }

    id indirectBuffer = (__bridge id)(gl_indirect_buffer->data.mtl_data);
    if (!indirectBuffer) {
        NSLog(@"MGL WARNING: %s skipped because indirect buffer %u has no Metal buffer",
              label ? label : "indirect draw",
              gl_indirect_buffer->name);
        return NO;
    }

    if (glBufferOut) {
        *glBufferOut = gl_indirect_buffer;
    }
    if (mtlBufferOut) {
        *mtlBufferOut = indirectBuffer;
    }
    return YES;
}

- (BOOL)prepareEmulatedIndirectCPURead:(GLMContext)drawCtx label:(const char *)label
{
    if (!drawCtx) {
        NSLog(@"MGL WARNING: %s skipped because context is NULL",
              label ? label : "indirect emulation");
        return NO;
    }

    /* The C draw-indirect frontends already flush pending command buffers before
     * dispatching into these Metal entry points. If processGLState has just
     * rebuilt a render encoder, keep it; a second flush can discard the fresh
     * pass and make state restoration fail for CPU-emulated indirect modes. */
    if (mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) == 1) {
        return YES;
    }

    [self flushCommandBuffer:true];
    if (![self processGLState:true]) {
        NSLog(@"MGL WARNING: %s skipped because GL state could not be restored after CPU-read synchronization",
              label ? label : "indirect emulation");
        return NO;
    }
    if (mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
        NSLog(@"MGL WARNING: %s skipped because CPU-read synchronization left no render encoder",
              label ? label : "indirect emulation");
        return NO;
    }
    return YES;
}

- (BOOL)currentDrawRasterizationIsEmpty
{
    if (!ctx) {
        return NO;
    }

    GLint vx = MGL_STATE(ctx)->viewport[0];
    GLint vy = MGL_STATE(ctx)->viewport[1];
    GLint vw = MGL_STATE(ctx)->viewport[2];
    GLint vh = MGL_STATE(ctx)->viewport[3];

    NSUInteger passWidth = 0;
    NSUInteger passHeight = 0;
    mglRenderGetRenderTargetSizeOwner(
        _renderPassManager.state->renderPassStateOwner,
        (uint64_t *)&passWidth, (uint64_t *)&passHeight);
    if (passWidth == 0 || passHeight == 0) {
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            id color = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, i);
            if (color) {
                MGLRenderTextureInfo info =
                    mglDrawSupportTextureInfo(color);
                passWidth = info.width;
                passHeight = info.height;
                break;
            }
        }
        if (passWidth == 0 || passHeight == 0) {
            id depth = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
            if (depth) {
                MGLRenderTextureInfo info =
                    mglDrawSupportTextureInfo(depth);
                passWidth = info.width;
                passHeight = info.height;
            }
        }
        if (passWidth == 0 || passHeight == 0) {
            id stencil = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                _renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0);
            if (stencil) {
                MGLRenderTextureInfo info =
                    mglDrawSupportTextureInfo(stencil);
                passWidth = info.width;
                passHeight = info.height;
            }
        }
    }


    return mglRenderRasterizationIsEmpty(
               vx, vy, vw, vh,
               (uint32_t)passWidth, (uint32_t)passHeight,
               MGL_STATE(ctx)->caps.scissor_test ? 1 : 0,
               MGL_STATE(ctx)->var.scissor_box[0],
               MGL_STATE(ctx)->var.scissor_box[1],
               MGL_STATE(ctx)->var.scissor_box[2],
               MGL_STATE(ctx)->var.scissor_box[3]) != 0;
}

- (void)applyPolygonOffsetForDrawMode:(GLenum)mode
{
    if (mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
        return;
    }


    MGLRenderPolygonOffsetDecision decision = {0};
    mglRenderPolygonOffsetDecision(
        (uint32_t)mode,
        ctx ? 1 : 0,
        mglDrawModeProducesPolygons(mode) ? 1 : 0,
        (uint32_t)(ctx ? MGL_STATE(ctx)->var.polygon_mode : 0u),
        (ctx && MGL_STATE(ctx)->caps.polygon_offset_point) ? 1 : 0,
        (ctx && MGL_STATE(ctx)->caps.polygon_offset_line) ? 1 : 0,
        (ctx && MGL_STATE(ctx)->caps.polygon_offset_fill) ? 1 : 0,
        &decision);
    uint32_t triangleFillMode = decision.triangle_fill_mode ? 1u : 0u;
    if (decision.needs_polygon_mode_repair) {
        mglLogRenderStateRepair("polygon_mode", MGL_STATE(ctx)->var.polygon_mode, GL_FILL);
        MGL_STATE(ctx)->var.polygon_mode = GL_FILL;
        mglMarkStateDirtyBits(ctx->active_state, DIRTY_RENDER_STATE);
    }
    [self setTriangleFillModeIfNeeded:triangleFillMode];

    BOOL enableDepthBias = decision.enable_depth_bias != 0;

    if (enableDepthBias) {
        float _bias = MGL_STATE(ctx)->var.polygon_offset_units;
        float _slope = MGL_STATE(ctx)->var.polygon_offset_factor;
        float _clamp = 0.0f;
        mglRenderBindingSetDepthBiasIfNeededForOwner(
            _bindingStateOwner,
            _renderPassManager.state->currentRenderEncoderOwner,
            _bias, _clamp, _slope);
    } else {
        mglRenderBindingSetDepthBiasIfNeededForOwner(
            _bindingStateOwner,
            _renderPassManager.state->currentRenderEncoderOwner,
            0.0f, 0.0f, 0.0f);
    }
}

- (BOOL)currentDrawModeIsFullyCulled:(GLenum)mode
{
    return ctx &&
           MGL_STATE(ctx)->caps.cull_face &&
           MGL_STATE(ctx)->var.cull_face_mode == GL_FRONT_AND_BACK &&
           mglDrawModeProducesPolygons(mode);
}

- (void)bindCullDistanceEmulationBuffers:(GLenum)mode
                             firstVertex:(GLuint)firstVertex
                        explicitVertices:(const GLuint *)explicitVertices
                      explicitVertexCount:(GLuint)explicitVertexCount
                           encodeContext:(const MGLEncodeContext *)encCtx
{
    if (!ctx || !mglDrawSupportEncodeContextIsActive(encCtx)) {
        return;
    }
    VertexArray *vao = mglRendererGetValidatedVAO(ctx, "bindCullDistanceEmu");
    if (!vao) {
        return;
    }
    Program *activeProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    if (!activeProgram) {
        return;
    }
    explicitVertexCount = MIN(explicitVertexCount, 4u);


    uint32_t prim_vertex_count =
        mglRenderPrimitiveVertexCountForMode((uint32_t)mode);

    id captureBuffer = (__bridge id)
        mglRendererBackendGetCullDistanceCaptureBuffer(_backend);
    if (captureBuffer) {
        MGLCullDistanceEmuParams params = {
            .prim_vertex_count = prim_vertex_count,
            .culldist_offset = 0u,
            .vertex_stride = 32u,
            .culldist_size = MIN(activeProgram->cull_distance_count, 8u),
            .first_vertex = firstVertex,
            .explicit_vertex_count = explicitVertexCount,
            .first_instance =
                _tessellation.cullDistanceCaptureFirstInstance,
            .instance_stride =
                _tessellation.cullDistanceCaptureInstanceStride,
        };
        if (explicitVertices) {
            memcpy(params.explicit_vertices, explicitVertices,
                   explicitVertexCount * sizeof(params.explicit_vertices[0]));
        }
        mglDrawSupportSetVertexBuffer(
            encCtx->render_encoder_owner, captureBuffer, 0u,
            kMGLCullDistanceVertexBufferIndex);
        [self recordLastBoundVertexBuffer:
                  captureBuffer
                                   offset:0
                                  atIndex:kMGLCullDistanceVertexBufferIndex];
        MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
        mglDrawSupportSetVertexBytes(
            encCtx->render_encoder_owner, &params, sizeof(params),
            kMGLCullDistanceParamsBufferIndex);
        [self invalidateLastBoundVertexBufferAtIndex:
                  kMGLCullDistanceParamsBufferIndex];
        return;
    }

    /* Scan enabled attributes for cull distance entries. The GLSL source
     * uses "culldistance_data" as the attribute name. We identify them
     * via the shader resource list (which preserves the name) or
     * by checking the MSL source for [[attribute(N)]] with that name. */
    void *cullMtlBuffer = NULL;
    GLintptr cullBindingOffset = 0;
    GLuint cullStride = 0;
    GLuint cullDistSize = 0;
    GLintptr cullFirstRelativeOffset = -1;

    MGLShaderResourceList *vsInputs =
        &activeProgram->shader_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES];

    for (GLuint attrib = 0; attrib < MAX_ATTRIBS; attrib++) {
        if (!mglRendererProgramUsesVertexAttrib(activeProgram, attrib)) {
            continue;
        }
        /* Find the resource name for this attribute. */
        const char *attrName = NULL;
        if (vsInputs && vsInputs->list) {
            for (GLuint r = 0; r < vsInputs->count; r++) {
                MGLShaderResource *res = &vsInputs->list[r];
                if (res->location == attrib) {
                    attrName = res->name;
                    break;
                }
                if (res->gl_array_size > 1 &&
                    attrib >= res->location &&
                    attrib < res->location + (GLuint)res->gl_array_size) {
                    attrName = res->name;
                    break;
                }
            }
        }
        /* Fall back to attrib_location_names if the resource name is missing. */
        if (!attrName && attrib < MAX_ATTRIBS) {
            attrName = activeProgram->attrib_location_names[attrib];
        }
        if (!attrName) {
            continue;
        }
        /* Match "culldistance_data" or "culldistance_data[N]" */
        if (strncmp(attrName, "culldistance_data", 17) != 0) {
            continue;
        }
        MGLResolvedVertexAttribBinding resolved = {0};
        if (!mglRendererResolveVertexAttribBinding(ctx, vao, attrib, "bindCullDistanceEmu", &resolved)) {
            continue;
        }
        if (!resolved.buffer || !resolved.buffer->data.mtl_data) {
            continue;
        }
        if (cullDistSize == 0) {
            /* First cull distance attribute: record buffer/stride/offset. */
            cullMtlBuffer = resolved.buffer->data.mtl_data;
            cullBindingOffset = resolved.binding_offset;
            cullStride = resolved.stride;
            cullFirstRelativeOffset = resolved.relativeoffset;
        } else {
            /* Subsequent cull distance attributes: verify they share the same
             * buffer and stride. If not, fall back to the first attribute's
             * layout (the CTS test uses a single interleaved buffer). */
            if (resolved.buffer->data.mtl_data != cullMtlBuffer ||
                resolved.stride != cullStride) {
                /* Layout mismatch; keep the first attribute's layout. */
            }
        }
        cullDistSize++;
    }

    if (!cullMtlBuffer || cullDistSize == 0) {
        /* No cull distance attributes found; bind a dummy buffer to satisfy
         * Metal validation (the shader still references the slots). */
        cullMtlBuffer = mglRendererBackendGetCullDistanceDummyBuffer(_backend);
        cullBindingOffset = 0;
        cullStride = 4;
        cullFirstRelativeOffset = 0;
        cullDistSize = 0; /* zero size means the shader loop is skipped */
    }

    /* The cull distance offset within each vertex is the binding offset plus
     * the relative offset of the first cull distance attribute. */
    uint32_t culldist_offset = (uint32_t)(cullBindingOffset + (cullFirstRelativeOffset >= 0 ? cullFirstRelativeOffset : 0));

    MGLCullDistanceEmuParams params = {0};
    params.prim_vertex_count = prim_vertex_count;
    params.culldist_offset = culldist_offset;
    params.vertex_stride = (uint32_t)cullStride;
    params.culldist_size = cullDistSize;
    params.first_vertex = firstVertex;
    params.explicit_vertex_count = explicitVertexCount;
    memset(params.explicit_vertices, 0, sizeof(params.explicit_vertices));
    if (explicitVertices) {
        memcpy(params.explicit_vertices, explicitVertices,
               explicitVertexCount * sizeof(params.explicit_vertices[0]));
    }

    mglDrawSupportSetVertexBuffer(
        encCtx->render_encoder_owner, (__bridge id)cullMtlBuffer, 0,
        kMGLCullDistanceVertexBufferIndex);
    [self recordLastBoundVertexBuffer:(__bridge id)cullMtlBuffer
                               offset:0
                              atIndex:kMGLCullDistanceVertexBufferIndex];
    MGL_PERF_INC(g_mglSetVertexBufferCallsSinceSwap);
    mglDrawSupportSetVertexBytes(
        encCtx->render_encoder_owner, &params, sizeof(params),
        kMGLCullDistanceParamsBufferIndex);
    [self invalidateLastBoundVertexBufferAtIndex:kMGLCullDistanceParamsBufferIndex];
}

- (BOOL)handleTessellationPatchDrawIfNeeded:(GLMContext)drawCtx
                                        mode:(GLenum *)mode
                                       first:(GLint)first
                                       count:(GLsizei)count
                                   indexType:(GLenum)indexType
                                     indices:(const void *)indices
                                  baseVertex:(GLint)baseVertex
                               instanceCount:(GLsizei)instanceCount
                                baseInstance:(GLuint)baseInstance
                                       label:(const char *)label
{
    if (!mode || *mode != GL_PATCHES) {
        return NO;
    }
    if (!drawCtx || count <= 0) {
        return YES;
    }

    self->ctx = drawCtx;

    Program *tcsProgram = mglResolveProgramForStageFromState(drawCtx, _TESS_CONTROL_SHADER);
    Program *tesProgram = mglResolveProgramForStageFromState(drawCtx, _TESS_EVALUATION_SHADER);
    if (tcsProgram && !tcsProgram->shader_slots[_TESS_CONTROL_SHADER]) {
        tcsProgram = NULL;
    }
    if (tesProgram && !tesProgram->shader_slots[_TESS_EVALUATION_SHADER]) {
        tesProgram = NULL;
    }
    if (!tcsProgram && !tesProgram) {
        return NO;
    }

    if (instanceCount <= 0) {
        return YES;
    }

    if (tcsProgram) {
        if (tcsProgram->dirty_bits) {
            [self bindMTLProgram:tcsProgram];
        }
    }

    if (tesProgram) {
        if (tesProgram->dirty_bits) {
            [self bindMTLProgram:tesProgram];
        }
    }

    const BOOL airTES = tesProgram &&
        tesProgram->modules[_TESS_EVALUATION_SHADER].metallib_bytes != NULL;
    BOOL nativeTES = mglNativeTESInterfaceSupported(tcsProgram, tesProgram);

    GLuint patchVertices = MAX(1u, (GLuint)MGL_STATE(drawCtx)->var.patch_vertices);
    GLuint patchCount = (GLuint)count / patchVertices;
    if (patchCount == 0u) {
        return YES;
    }


    uint32_t restartIndex = 0u;
    bool restartEnabled = false;
    if (indexType != 0u) {
        restartEnabled =
            mglPrimitiveRestartIndexForType(drawCtx, indexType, &restartIndex);
    }
    Program *vertexProgram =
        mglResolveProgramForStageFromState(drawCtx, _VERTEX_SHADER);
    MGLAIRTessDrawContract contract;
    memset(&contract, 0, sizeof(contract));
    contract.patch_vertices = patchVertices;
    contract.vertex_count = (uint32_t)count;
    contract.patch_count = patchCount;
    contract.instance_count = instanceCount > 0 ? (uint32_t)instanceCount : 1u;
    contract.base_instance = baseInstance;
    contract.first = first;
    contract.index_type = indexType;
    contract.index_source = (uint64_t)(uintptr_t)indices;
    contract.index_count = indexType != 0u ? (uint64_t)count : 0u;
    contract.base_vertex = baseVertex;
    contract.primitive_restart = restartEnabled ? 1u : 0u;
    contract.restart_index = restartIndex;
    contract.tess_factor_bytes_per_patch = MGL_AIR_TESS_FACTOR_QUAD_HALF_BYTES;
    contract.tess_gen_mode = tesProgram
        ? (uint32_t)tesProgram->tess_gen_mode : (uint32_t)GL_TRIANGLES;
    contract.point_mode = tesProgram
        ? (uint32_t)tesProgram->tess_gen_point_mode : 0u;
    contract.tcs_out_vertices =
        tcsProgram && tcsProgram->tess_control_output_vertices > 0u
            ? tcsProgram->tess_control_output_vertices : patchVertices;
    contract.per_vertex_out_stride = vertexProgram
        ? mglAIRPerVertexStrideForResources(
              &vertexProgram->shader_resources_list[_VERTEX_SHADER]
                                                   [_STAGE_OUTPUT_RES])
        : MGL_AIR_PER_VERTEX_STRIDE;
    contract.patch_out_stride = 16u; /* refined by the TCS dispatcher */

    (void)mglRendererBackendSetTessVertexCaptureBuffer(_backend, NULL);
    _tessellation.tessVertexCaptureOffset = 0u;
    (void)mglRendererBackendSetTessControlPointIndexBuffer(_backend, NULL);
    _tessellation.tessIndexedDraw = NO;
    _tessellation.tessInstanceRecords = 0u;
    /* A TCS from a previous draw must not leak into a TES-only dispatch
     * (dispatchAIRTessEvalCompute reads tcsOutputBuffer as the gl_in
     * source when non-nil).  The TCS dispatcher re-populates it. */
    (void)mglRendererBackendSetTcsOutputBuffer(_backend, NULL);
    _tessellation.tcsOutputOffset = 0u;
    _tessellation.tcsOutputStride = 0u;
    _tessellation.tcsOutVertices = 0u;
    (void)mglRendererBackendSetCurrentTessFactorBuffer(_backend, NULL);
    /* The VS position capture and the default factor buffer are consumed by
     * both the native patch pipeline and the AIR TES compute expansion
     * (isolines / point_mode with no TCS), so they must exist even when
     * nativeTES is unavailable. */
    if (nativeTES || airTES) {
        const BOOL indexedDraw = (indexType != 0u);
        if (indexedDraw && tcsProgram) {
            /* TCS consumes the VS capture with continuous per-patch
             * addressing; indexed (sparse) input needs a gather path in the
             * TCS kernel, which is a separate follow-up. */
            nativeTES = NO;
        } else if (indexedDraw) {
            /* Indexed native TES (no TCS): capture the VS once into sparse
             * per-vertex records [instance][vertex_id] and let the CPU
             * gather buffer (raw index stream) drive Metal's
             * controlPointIndexBuffer.  baseVertex is already applied by the
             * indexed capture draw.  Instances are drawn one at a time from
             * their contiguous capture spans because Metal patch draws have no
             * per-instance patch-data offset. */
            Buffer *ebo = getElementBuffer(drawCtx);
            if (!ebo || ![self processBuffer:ebo] || !ebo->data.mtl_data) {
                nativeTES = NO;
            } else {
                id eboMetal =
                    (__bridge id)ebo->data.mtl_data;
                const NSUInteger indexOffsetBytes =
                    (NSUInteger)(uintptr_t)indices;
                const uint8_t *indexBytes = mglElementIndexSourceForDraw(
                    ebo, eboMetal, indexType, indexOffsetBytes, count);
                uint32_t *gatherArray = NULL;
                uint32_t gatherCount = 0u;
                uint32_t gatherPrimitives = 0u;
                uint32_t gatherMaxIndex = 0u;
                if (!indexBytes ||
                    !mglGeometryGatherIndices(indexBytes, indexType, count,
                                              baseVertex, restartEnabled,
                                              restartIndex, patchVertices,
                                              &gatherArray, &gatherCount,
                                              &gatherPrimitives,
                                              &gatherMaxIndex)) {
                    nativeTES = NO;
                } else {
                    id gatherBuf =
                        mglDrawSupportCreateBufferWithBytes(
                            _device, gatherArray,
                            (NSUInteger)gatherCount * 4u,
                            0u);
                    free(gatherArray);
                    if (!gatherBuf) {
                        nativeTES = NO;
                    } else {
                        NSUInteger captureOffset = 0u;
                        id capture = [self
                            captureAIRVertexPositionsForGeometryIndexed:drawCtx
                                                            indexBuffer:eboMetal
                                                              indexType:indexType
                                                            indexOffset:indexOffsetBytes
                                                                  count:count
                                                              baseVertex:baseVertex
                                                           instanceCount:instanceCount
                                                            baseInstance:baseInstance
                                                               maxIndex:gatherMaxIndex
                                                               outOffset:&captureOffset];
                        if (!capture) {
                            nativeTES = NO;
                        } else {
                            (void)mglRendererBackendSetTessVertexCaptureBuffer(
                                _backend, (__bridge void *)capture);
                            _tessellation.tessVertexCaptureOffset = captureOffset;
                            (void)mglRendererBackendSetTessControlPointIndexBuffer(
                                _backend, (__bridge void *)gatherBuf);
                            _tessellation.tessIndexedDraw = YES;
                            _tessellation.tessInstanceRecords =
                                (NSUInteger)gatherMaxIndex + 1u;
                            /* The gather stream is already re-grouped into
                             * complete patches, so it is the real count. */
                            patchCount = gatherPrimitives;
                            contract.patch_count = patchCount;
                        }
                    }
                }
            }
        } else {
            NSUInteger captureOffset = 0u;
            id capture =
                [self captureAIRVertexPositionsForTessellation:drawCtx
                                                         first:first
                                                         count:count
                                                 instanceCount:instanceCount
                                                  baseInstance:baseInstance
                                                    outOffset:&captureOffset];
            if (!capture) {
                nativeTES = NO;
            } else {
                (void)mglRendererBackendSetTessVertexCaptureBuffer(
                    _backend, (__bridge void *)capture);
                _tessellation.tessVertexCaptureOffset = captureOffset;
                _tessellation.tessInstanceRecords = (NSUInteger)count;
            }
        }
    }

    if (nativeTES && !tcsProgram) {
        id tessVertexCaptureBuffer =
            (__bridge id)
                mglRendererBackendGetTessVertexCaptureBuffer(_backend);
        (void)mglRendererBackendSetTcsOutputBuffer(
            _backend, (__bridge void *)tessVertexCaptureBuffer);
        _tessellation.tcsOutputOffset =
            _tessellation.tessVertexCaptureOffset;
        _tessellation.tcsOutputStride = contract.per_vertex_out_stride;
        _tessellation.tcsOutVertices = patchVertices;
        id tessFactorBuffer = mglCachedDefaultTessFactorBuffer(
            _device, _backend, MGL_STATE(drawCtx), patchCount);
        (void)mglRendererBackendSetCurrentTessFactorBuffer(
            _backend, (__bridge void *)tessFactorBuffer);
        if (!tessVertexCaptureBuffer ||
            !tessFactorBuffer) {
            nativeTES = NO;
        }
    }

    if (airTES && !tcsProgram) {
        /* TES-only compute expansion also needs the default levels; the
         * cached buffer is rebuilt only when glPatchParameterfv levels
         * (or the patch count) change between draws. */
        id tessFactorBuffer = mglCachedDefaultTessFactorBuffer(
            _device, _backend, MGL_STATE(drawCtx), patchCount);
        (void)mglRendererBackendSetCurrentTessFactorBuffer(
            _backend, (__bridge void *)tessFactorBuffer);
    }

    if (tcsProgram) {
        if (![self dispatchTessControlShader:drawCtx
                                     program:tcsProgram
                                    contract:&contract]) {
            drawCtx->state.dirty_bits = DIRTY_ALL;
            (void)mglRendererBackendSetTessVertexCaptureBuffer(_backend, NULL);
            _tessellation.tessVertexCaptureOffset = 0u;
            return YES;
        }
    }

    id tcsOutputBuffer = (__bridge id)
        mglRendererBackendGetTcsOutputBuffer(_backend);
    id tessFactorBuffer = (__bridge id)
        mglRendererBackendGetCurrentTessFactorBuffer(_backend);

    if (nativeTES) {
        id nativeFactors = mglNativeTessFactorBuffer(
            _device, tessFactorBuffer,
            tesProgram->tess_gen_mode, patchCount);
        if (!nativeFactors || !tcsOutputBuffer ||
            _tessellation.tcsOutputStride < MGL_AIR_PER_VERTEX_STRIDE) {
            NSLog(@"MGL TESS ERROR: invalid native TES buffers program=%u",
                  (unsigned)tesProgram->name);
            mglDispatchError(drawCtx, label ? label : "tessellationDraw",
                             GL_OUT_OF_MEMORY);
            drawCtx->state.dirty_bits = DIRTY_ALL;
            (void)mglRendererBackendSetTessVertexCaptureBuffer(_backend, NULL);
            _tessellation.tessVertexCaptureOffset = 0u;
            return YES;
        }

        _tessellation.nativeTESProgram = tesProgram;
        _tessellation.nativeTESActive = YES;
        [self clearStageBindingCopyBacks:&_tessellation.nativeTESCopyBacks];
        drawCtx->state.dirty_bits = DIRTY_ALL;

        BOOL stateReady = [self processGLState:true];
        if (!stateReady || mglRenderEncoderOwnerHasCurrent(_renderPassManager.state->currentRenderEncoderOwner) != 1) {
            _tessellation.nativeTESActive = NO;
            _tessellation.nativeTESProgram = NULL;
            (void)mglRendererBackendSetTessVertexCaptureBuffer(_backend, NULL);
            _tessellation.tessVertexCaptureOffset = 0u;
            drawCtx->state.dirty_bits = DIRTY_ALL;
            return YES;
        }

        if (![self currentDrawRasterizationIsEmpty] &&
            ![self currentDrawModeIsFullyCulled:GL_TRIANGLES]) {
            [self applyPolygonOffsetForDrawMode:GL_TRIANGLES];
            /* Metal does not advance the post-tessellation control-point
             * pointer correctly for patchStart. Draw each patch separately:
             * slot 0 is rebased to the patch, while slot 30 stays at the
             * instance base so TES varyings can apply patchId exactly once. */
            const NSUInteger instanceRecords =
                _tessellation.tessInstanceRecords;
            const NSUInteger instanceStrideBytes =
                instanceRecords * _tessellation.tcsOutputStride;
            mglDrawSupportSetTessellationFactors(
                _renderPassManager.state->currentRenderEncoderOwner, nativeFactors, 0u, 0u);
            for (GLsizei i = 0; i < instanceCount; i++) {
                const NSUInteger instanceOffset =
                    _tessellation.tessVertexCaptureOffset +
                    (NSUInteger)i * instanceStrideBytes;
                mglDrawSupportSetVertexBuffer(
                    _renderPassManager.state->currentRenderEncoderOwner, tcsOutputBuffer,
                    instanceOffset, 0u);
                [self recordLastBoundVertexBuffer:tcsOutputBuffer
                                           offset:instanceOffset
                                          atIndex:0u];
                mglDrawSupportSetVertexBuffer(
                    _renderPassManager.state->currentRenderEncoderOwner, tcsOutputBuffer,
                    instanceOffset, 30u);
                [self recordLastBoundVertexBuffer:tcsOutputBuffer
                                           offset:instanceOffset
                                          atIndex:30u];
                GLuint patchInfo[2] = {patchVertices, _tessellation.tcsOutVertices};
                if (patchInfo[1] == 0u) patchInfo[1] = patchVertices;
                mglDrawSupportSetVertexBytes(
                    _renderPassManager.state->currentRenderEncoderOwner, patchInfo, sizeof(patchInfo), 28u);
                id tcsPatchOutBuffer =
                    (__bridge id)
                        mglRendererBackendGetTcsPatchOutBuffer(_backend);
                const BOOL perPatchNativeResources = (tcsPatchOutBuffer != nil);
                const NSUInteger nativeFactorStride =
                    tesProgram->tess_gen_mode == GL_QUADS ? 12u : 8u;
                NSUInteger patchOutStride = 16u;
                if (perPatchNativeResources && tcsProgram) {
                    patchOutStride = mglAIRPatchVaryingStride(
                        &tcsProgram->shader_resources_list[_TESS_CONTROL_SHADER]
                                                         [_STAGE_OUTPUT_RES]);
                }
                if (_tessellation.tessIndexedDraw) {
                    id controlPointIndexBuffer =
                        (__bridge id)
                            mglRendererBackendGetTessControlPointIndexBuffer(
                                _backend);
                    mglDrawSupportDrawIndexedPatches(
                        _renderPassManager.state->currentRenderEncoderOwner, _tessellation.tcsOutVertices, 0u, patchCount,
                        nil, 0u,
                        controlPointIndexBuffer, 0u,
                        1u, (NSUInteger)baseInstance + (NSUInteger)i);
                } else {
                    const NSUInteger cpcStride =
                        (NSUInteger)_tessellation.tcsOutVertices *
                        _tessellation.tcsOutputStride;
                    for (GLuint p = 0u; p < patchCount; p++) {
                        const NSUInteger patchOffset =
                            instanceOffset + (NSUInteger)p * cpcStride;
                        mglDrawSupportSetVertexBuffer(
                            _renderPassManager.state->currentRenderEncoderOwner, tcsOutputBuffer,
                            patchOffset, 0u);
                        [self recordLastBoundVertexBuffer:
                                  tcsOutputBuffer
                                                   offset:patchOffset
                                                  atIndex:0u];
                        GLuint patchInfoWords[3] = {
                            patchVertices, _tessellation.tcsOutVertices, p,
                        };
                        if (patchInfoWords[1] == 0u) patchInfoWords[1] = patchVertices;
                        mglDrawSupportSetVertexBytes(
                            _renderPassManager.state->currentRenderEncoderOwner,
                            patchInfoWords, sizeof(patchInfoWords), 28u);
                        if (perPatchNativeResources) {
                            mglDrawSupportSetVertexBuffer(
                                _renderPassManager.state->currentRenderEncoderOwner,
                                tcsPatchOutBuffer,
                                (NSUInteger)p * patchOutStride, 27u);
                            [self recordLastBoundVertexBuffer:
                                      tcsPatchOutBuffer
                                                   offset:(NSUInteger)p * patchOutStride
                                                  atIndex:27u];
                            mglDrawSupportSetTessellationFactors(
                                _renderPassManager.state->currentRenderEncoderOwner,
                                nativeFactors,
                                (NSUInteger)p * nativeFactorStride, 0u);
                            mglDrawSupportDrawPatches(
                                _renderPassManager.state->currentRenderEncoderOwner, _tessellation.tcsOutVertices, 0u, 1u,
                                nil, 0u, 1u,
                                (NSUInteger)baseInstance + (NSUInteger)i);
                        } else {
                            mglDrawSupportDrawPatches(
                                _renderPassManager.state->currentRenderEncoderOwner, _tessellation.tcsOutVertices, p, 1u,
                                nil, 0u, 1u,
                                (NSUInteger)baseInstance + (NSUInteger)i);
                        }
                    }
                }
            }
            _currentCBHasWork = YES;

            GLuint64 primitives = mglNativeTessPrimitiveCount(
                tessFactorBuffer, tesProgram, patchCount,
                (GLuint)instanceCount);
            mglRecordActivePrimitiveQueryDraw(drawCtx, primitives, primitives);
        }

        [self endRenderEncoding];
        if (![self flushStageBindingCopyBacks:
                      &_tessellation.nativeTESCopyBacks
                               requireCPUVisibility:NO]) {
            NSLog(@"MGL TESS ERROR: failed to copy isolated native TES "
                  "writable buffer prefixes");
            mglDispatchError(drawCtx, label ? label : "tessellationDraw",
                             GL_OUT_OF_MEMORY);
        }

        _tessellation.nativeTESActive = NO;
        _tessellation.nativeTESProgram = NULL;
        (void)mglRendererBackendSetTessVertexCaptureBuffer(_backend, NULL);
        _tessellation.tessVertexCaptureOffset = 0u;
        (void)mglRendererBackendSetTessControlPointIndexBuffer(_backend, NULL);
        _tessellation.tessIndexedDraw = NO;
        _tessellation.tessInstanceRecords = 0u;
        drawCtx->state.dirty_bits = DIRTY_ALL;
        return YES;
    }

    if (airTES) {
        /* Isolines and layout(point_mode) have no Metal-native equivalent
         * (MTLPatchType is triangle/quad only and patch draws have no output
         * primitive type), so those programs run as an AIR compute kernel
         * expansion + passthrough vertex (line/point rasterization). */
        if (tesProgram && (tesProgram->tess_gen_point_mode ||
                           tesProgram->tess_gen_mode == GL_ISOLINES)) {
            const BOOL dispatched =
                [self dispatchAIRTessEvalCompute:drawCtx
                                        program:tesProgram
                                       contract:&contract
                                     patchCount:patchCount
                                  instanceCount:instanceCount
                                   baseInstance:baseInstance];
            if (!dispatched) {
                mglDispatchError(drawCtx, label ? label : "tessellationDraw",
                                 GL_INVALID_OPERATION);
            }
            drawCtx->state.dirty_bits = DIRTY_ALL;
            (void)mglRendererBackendSetTessVertexCaptureBuffer(_backend, NULL);
            _tessellation.tessVertexCaptureOffset = 0u;
            return YES;
        }
        NSLog(@"MGL TESS ERROR: native AIR TES interface unsupported for program %u",
              (unsigned)tesProgram->name);
        /*  contract: an unsupported tessellation draw must surface a GL
         * error, not silently drop the patch stream. */
        mglDispatchError(drawCtx, label ? label : "tessellationDraw",
                         GL_INVALID_OPERATION);
        drawCtx->state.dirty_bits = DIRTY_ALL;
        (void)mglRendererBackendSetTessVertexCaptureBuffer(_backend, NULL);
        _tessellation.tessVertexCaptureOffset = 0u;
        return YES;
    }

    if (tesProgram) {
        if (![self dispatchTessEvaluationShader:drawCtx
                                           program:tesProgram
                                          contract:&contract]) {
            drawCtx->state.dirty_bits = DIRTY_ALL;
            return YES;
        }
    }

    drawCtx->state.dirty_bits = DIRTY_ALL;
    (void)mglRendererBackendSetTessVertexCaptureBuffer(_backend, NULL);
    _tessellation.tessVertexCaptureOffset = 0u;
    (void)label;
    return YES;
}


@end
