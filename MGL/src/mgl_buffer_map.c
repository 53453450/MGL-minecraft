/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_buffer_map.c — MGLRenderer+Buffer.m moved here (P0-1, log 102).
 *
 * Nine category methods, three C helpers they carried, and two of the file's C
 * functions (mglAdvanceFrameGeneration / mglRecordFrameCompleted /
 * mglNoteBufferEncoded) are C now.  What each translation needed:
 *
 *   ctx                        -> areas.ctx
 *   MGL_STATE(ctx)             -> mglBufferMapState(&areas) (the dual proxy)
 *   _tessellation.nativeTESActive
 *                              -> areas.tess_native_tes_active
 *   _pipelineCache.state->…    -> areas.pipeline_cache->…
 *   [self …]                   -> the C entry points below
 *   NSLog                      -> fprintf(stderr, …) on the same sink
 *   NSUInteger                 -> size_t / unsigned long
 *   __bridge_transfer id       -> a +1 void * the caller releases
 *
 * Two dead functions were dropped rather than translated: see the note in
 * mgl_buffer_map.h.
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <CoreFoundation/CoreFoundation.h>

#include "mgl_buffer_map.h"
#include "mgl_renderer_ports.h"        /* state areas */
#include "mgl_render.h"                /* the mglRender* facade */
#include "mgl_renderer_backend.h"      /* mglRendererGetProgramBindingCount */
#include "mgl_vertex_attrib_plan.h"    /* mglRenderPlanVertexAttribBuffers */
#include "mgl_vertex_attrib_query.h"   /* mglRendererGetValidatedVAO */
#include "mgl_vertex_format.h"         /* MAX_ATTRIBS */
#include "mgl_binding_policy.h"        /* mglRenderStageMapsVertexAttribs */
#include "mgl_sampler_compat.h"        /* mglRendererResourceLooksSamplerLike */
#include "mgl_trace_strategy.h"        /* trace-log gates */
#include "mgl_state_compat.h"          /* mglShouldLogSmallBaseBinding */
#include "mgl_program_resource.h"      /* mglShaderStageName */
#include "mgl_shader_resource.h"       /* mglMGLShaderResourceTypeName */
#include "mgl_batch_mtl_encode.h"      /* mglNoteBufferEncoded */
#include "hash_table.h"                /* searchHashTable */
#include "mgl_trace_log.h"             /* mglTraceLog / kMGLDiagnosticStateLogs */

/* === C prototypes for functions defined in Objective-C translation units ===
 * Their declarations live in MGLRenderer+Draw_Private.h, which a .c file
 * cannot include, so they are repeated here the way mgl_renderer_ports.c and
 * framebuffers.c already do.  NSUInteger is `unsigned long` on every target
 * MGL builds for. */
extern Buffer *mglRendererGetValidatedBuffer(GLMContext ctx, Buffer *candidate,
                                             const char *where,
                                             unsigned long slot);
extern bool mglRenderCheckForDirtyBufferData(GLMContext ctx,
                                             BufferMapList *buffer_map_list,
                                             const char *where);
extern bool mglRenderUpdateDirtyBaseBufferList(GLMContext ctx,
                                               BufferMapList *buffer_map_list,
                                               const char *where);
extern int mglRenderVertexBufferIndexForAttribute(GLMContext ctx, GLMState *state,
                                                  int attribute,
                                                  const char *where);

/* The `where` labels the Objective-C methods used to pass (their __FUNCTION__),
 * kept verbatim so the diagnostics that print them stay identical. */
#define MGL_BUFFER_MAP_WHERE_CHECK_DIRTY \
    "-[MGLRenderer(Buffer) checkForDirtyBufferData:]"
#define MGL_BUFFER_MAP_WHERE_UPDATE_DIRTY \
    "-[MGLRenderer(Buffer) updateDirtyBaseBufferList:]"
#define MGL_BUFFER_MAP_WHERE_VERTEX_INDEX \
    "-[MGLRenderer(Buffer) getVertexBufferIndexWithAttributeSet:]"

/* === helpers ============================================================ */

/* MGL_STATE() from MGLRenderer_Private.h, expressed in C: the core state's
 * active pointer wins, NULL means "the context's own state". */
static GLMState *mglBufferMapState(const MGLRendererStateAreas *areas)
{
    if (areas->core && areas->core->activeState) {
        return areas->core->activeState;
    }
    return areas->ctx ? areas->ctx->active_state : NULL;
}

/* C twin of the Objective-C inline in MGLRenderer+Draw_Private.h (same policy,
 * and the same compile-time-off diagnostic gate). */
static inline bool mglBufferMapShouldTraceCall(uint64_t count)
{
    if (!kMGLDiagnosticStateLogs) {
        return false;
    }
    return (count <= 80ull) || ((count % 500ull) == 0ull);
}

/* Releases the +1 reference mglBufferCreateConvertedVertexBufferForAttribKind
 * hands back.  It is a cache retain, not a create, so it must NOT go through
 * mglSafeReleaseMetalObj: that would count a release for an object whose create
 * was counted once when the cache built it.  This is what ARC did for the same
 * reference (a plain release at scope exit) before the file moved to C. */
void mglBufferReleaseConvertedVertexBuffer(void *buffer)
{
    if (buffer) {
        CFRelease((CFTypeRef)buffer);
    }
}

/* === vertex attribute conversion ======================================== */

/* +1 converted buffer, or NULL.  The conversion itself is C++ (the cache owns
 * the object; this reference is a retain). */
static void *mglBufferCreateConvertedVertexBuffer(
    Buffer *sourceBuffer,
    const MGLResolvedVertexAttribBinding *resolved,
    const MGLRenderVertexConversion *base,
    size_t *outStride)
{
    MGLRenderVertexConversion conversion = {0};
    if (base) {
        conversion = *base;
    }
    conversion.binding_offset = resolved ? resolved->binding_offset : -1;
    conversion.relative_offset = resolved ? resolved->relativeoffset : -1;
    conversion.stride = resolved ? resolved->stride : 0u;

    uint64_t convertedStride = 0;
    void *convertedBuffer = NULL;
    char error[256] = {0};
    if (mglRenderConvertVertexBuffer(
            sourceBuffer, &conversion, &convertedStride, &convertedBuffer,
            error, sizeof(error)) != 0 || !convertedBuffer) {
        fprintf(stderr,
                "MGL BUFFER ERROR: Metal-cpp vertex conversion failed buffer=%u kind=%u: %s\n",
                sourceBuffer ? sourceBuffer->name : 0u,
                (unsigned)conversion.kind,
                error[0] ? error : "?");
        return NULL;
    }
    if (outStride) {
        *outStride = (size_t)convertedStride;
    }
    return convertedBuffer;
}

void *mglBufferCreateConvertedVertexBufferForAttribKind(
    void *renderer, int attrib_kind, Buffer *sourceBuffer,
    const MGLResolvedVertexAttribBinding *resolved, uint32_t component_count,
    uint32_t type, int normalized, int dst_is_int, size_t *outStride)
{
    (void)renderer;
    if (outStride) {
        *outStride = 0;
    }
    if (!sourceBuffer || !resolved) {
        return NULL;
    }
    MGLRenderVertexConversion conversion = {0};
    if (mglRenderFillVertexConversionFromAttribKind(
            attrib_kind, component_count, type,
            normalized ? 1 : 0, dst_is_int ? 1 : 0, &conversion) != 0) {
        return NULL;
    }
    return mglBufferCreateConvertedVertexBuffer(sourceBuffer, resolved,
                                                &conversion, outStride);
}

/* Resolver seam for the buffer-plan layer: the plan decides how resolved
 * attribute bindings group into Metal vertex buffer slots, while resolving one
 * attribute against live GL state stays here (it needs the context's validated
 * buffer table). */
typedef struct MGLVertexAttribPlanResolveCtx_t {
    GLMContext ctx;
    VertexArray *vao;
} MGLVertexAttribPlanResolveCtx;

static int mglResolveVertexAttribForPlan(void *user, GLuint attribute,
                                         MGLResolvedVertexAttribBinding *out)
{
    const MGLVertexAttribPlanResolveCtx *resolveCtx =
        (const MGLVertexAttribPlanResolveCtx *)user;
    if (!resolveCtx) {
        return 1;
    }
    return mglRendererResolveVertexAttribBinding(
               resolveCtx->ctx, resolveCtx->vao, attribute,
               "mapGLBuffersToMTLBufferMap", out)
               ? 0
               : 1;
}

/* Acquire renderer-owned packed struct storage from the C++ backend. */
static Buffer *mglGetPackedStructBuffer(const void *data,
                                        size_t size)
{
    char error[256] = {0};
    Buffer *buffer = mglRenderAcquirePackedStructBuffer(
        data, size, error, sizeof(error));
    if (!buffer) {
        fprintf(stderr, "MGL ERROR: Metal-cpp packed struct buffer failed: %s\n",
                error[0] ? error : "unknown");
    }
    return buffer;
}

/* === buffer mapping ===================================================== */

bool mglRendererMapGLBuffersToMTLBufferMap(void *renderer,
                                           BufferMapList *buffer_map, int stage)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;

    static uint64_t s_mapCallCountByStage[8] = {0};
    uint64_t mapCall = 0;
    if (stage >= 0 && stage < 8) {
        mapCall = ++s_mapCallCountByStage[stage];
    } else {
        mapCall = ++s_mapCallCountByStage[0];
    }

    if (kMGLDiagnosticStateLogs && mglBufferMapShouldTraceCall(mapCall)) {
        mglTraceLog("MGL TRACE map.begin stage=%d call=%llu preCount=%u program=%u",
                    stage,
                    (unsigned long long)mapCall,
                    buffer_map ? buffer_map->count : 0,
                    ctx ? (unsigned)mglBufferMapState(&areas)->program_name : 0u);
    }

    /* init mapped buffer count */
    buffer_map->count = 0;

    if (!mglRendererMapShaderBufferResourcesToBufferMap(renderer, buffer_map,
                                                        stage)) {
        return false;
    }

    /* bind vao attribs to buffers (attribs can share the same buffer) */
    if (mglRenderStageMapsVertexAttribs(stage))
    {
        const int count = mglRendererGetProgramBindingCount(ctx, stage, _STAGE_INPUT_RES);
        VertexArray *vao = mglRendererGetValidatedVAO(ctx, "mapGLBuffersToMTLBufferMap");
        if (!vao) {
            if (count > 0) {
                fprintf(stderr,
                        "MGL WARNING: mapGLBuffersToMTLBufferMap: stage inputs=%d but VAO is invalid/null, skipping attrib mapping\n",
                        count);
            }
        } else {
            /* Candidate attributes: enabled in the VAO (or every attribute when
             * the VAO carries no explicit enable mask) and consumed by the
             * program's vertex stage.  The grouping/limits themselves are the
             * buffer-plan layer's job (mglRenderPlanVertexAttribBuffers). */
            const bool explicitAttribMask = (vao->enabled_attribs != 0u);
            Program *vertexProgram =
                mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
            uint32_t candidateMask = 0u;
            for (GLuint att = 0u; att < MAX_ATTRIBS; att++) {
                if (explicitAttribMask &&
                    (vao->enabled_attribs & (0x1u << att)) == 0u) {
                    continue;
                }
                if (!mglRendererProgramUsesVertexAttrib(vertexProgram, att)) {
                    continue;
                }
                candidateMask |= (0x1u << att);
            }
            MGLVertexAttribPlanResolveCtx resolveCtx = {ctx, vao};
            MGLVertexAttribBufferPlanInput planInput = {0};
            planInput.candidate_mask = candidateMask;
            planInput.stage_input_count = count;
            planInput.stage = stage;
            planInput.map_capacity = MAX_MAPPED_BUFFERS;
            planInput.pipeline_state =
                (const void *)(areas.pipeline_cache
                                   ? areas.pipeline_cache->pipelineState
                                   : NULL);
            Buffer *drawIndexBuffer = vao->element_array.buffer;
            planInput.index_buffer_metal =
                drawIndexBuffer ? drawIndexBuffer->data.mtl_data : NULL;
            planInput.vao = (const void *)vao;
            planInput.resolve = mglResolveVertexAttribForPlan;
            planInput.resolve_user = (void *)&resolveCtx;
            if (mglRenderPlanVertexAttribBuffers(buffer_map, &planInput) != 0) {
                return false;
            }
        }
    }

    if (kMGLDiagnosticStateLogs && mglBufferMapShouldTraceCall(mapCall)) {
        mglTraceLog("MGL TRACE map.end stage=%d call=%llu mappedCount=%u",
                    stage,
                    (unsigned long long)mapCall,
                    buffer_map ? buffer_map->count : 0);
    }

    return true;
}

/* Map shader buffer resources from the cached buffer binding plan.  Called only
 * with a valid stage plan (see mglRendererMapShaderBufferResourcesToBufferMap,
 * which owns plan availability); returns false when a resource cannot be
 * bound. */
bool mglRendererMapShaderBufferResourcesViaPlan(
    void *renderer, BufferMapList *buffer_map, int stage, Program *program,
    const MGLStageBufferPlan *stagePlan)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    GLMState *state = mglBufferMapState(&areas);

    if (!program || !stagePlan || !stagePlan->valid || !buffer_map) {
        return false;
    }

    for (uint32_t pi = 0; pi < stagePlan->entry_count; pi++)
    {
        const MGLBufferPlanEntry *entry = &stagePlan->entries[pi];
        int spvc_type = (int)entry->resource_type;

        /* Validate the resource is still in range (plan was built from the
         * same list, but guard against any unexpected reallocation). */
        if (!mglRenderShaderResourceIndexValid(
                spvc_type, entry->resource_index,
                program->shader_resources_list[stage][spvc_type].count)) {
            return false;  /* fall back to original path */
        }
        MGLShaderResource *resource =
            &program->shader_resources_list[stage][spvc_type].list[entry->resource_index];

        /* Resolve buffer arrays (same logic as the original path). */
        int gl_buffer_type = mglRenderShaderResourceToGLBufferType(spvc_type);
        if (gl_buffer_type < 0) {
            return false;
        }

        BufferBaseTarget *buffers;
        BufferBaseTarget *fallbackBuffers = NULL;
        if (mglRenderUsePlainUniformBuffers(spvc_type)) {
            buffers = program->plain_uniform_buffers;
            fallbackBuffers = state->buffer_base[gl_buffer_type].buffers;
        } else {
            buffers = state->buffer_base[gl_buffer_type].buffers;
        }

        /* MGL_DEBUG_STRUCT_PACK diagnostic (gated by getenv). */
        if (mglRenderUsePlainUniformBuffers(spvc_type) &&
            getenv("MGL_DEBUG_STRUCT_PACK")) {
            fprintf(stderr,
                    "MGL STRUCTCHECK program=%u stage=%d name=%s ubo_members=%p count=%u req_size=%lu samplerLike=%d unifLoc=%d\n",
                    (unsigned)program->name, stage,
                    resource->name ? resource->name : "(null)",
                    (void *)resource->ubo_members,
                    (unsigned)resource->ubo_member_count,
                    (unsigned long)resource->required_size,
                    mglRendererResourceLooksSamplerLike(resource, spvc_type) ? 1 : 0,
                    resource->uniform_location);
        }

        /* ---- Struct packing path (plain uniform structs) ---- */
        if (mglRenderBufferPlanIsStructPacked(entry->flags))
        {
            GLuint loc_step = entry->loc_step;
            GLint base_loc = entry->base_loc;
            GLuint struct_size = entry->struct_size;
            GLuint array_size = entry->element_count;
            bool allowFallback = mglRenderBufferPlanAllowFallback(
                fallbackBuffers ? 1 : 0, entry->flags) != 0;

            for (GLuint element = 0; element < array_size; element++) {
                GLuint metal_binding = mglBufferPlanMetalBindingForElement(entry, element);
                GLuint elem_loc_start = element * loc_step;
                GLuint elem_loc_end = (element + 1u) * loc_step;
                GLuint elem_byte_start = element * (GLuint)struct_size;

                uint8_t stack_packed[256];
                uint8_t *packed = (struct_size <= sizeof(stack_packed))
                                  ? stack_packed
                                  : (uint8_t *)calloc(1, struct_size);
                if (!packed) continue;
                memset(packed, 0, struct_size);

                for (GLuint m = 0; m < entry->struct_member_count; m++) {
                    const MGLBufferPlanStructMember *sm = &entry->struct_members[m];

                    GLuint member_loc_off = sm->member_loc_off;
                    if (!mglRenderStructMemberInElementRange(
                            member_loc_off, elem_loc_start, elem_loc_end)) {
                        continue;
                    }

                    GLuint member_offset = mglRenderMemberOffsetInElement(
                        sm->member_offset_in_elem, elem_byte_start);
                    if (!mglRenderMemberOffsetInStruct(member_offset,
                                                       (uint32_t)struct_size)) {
                        continue;
                    }

                    GLint member_loc = sm->member_loc;
                    if (!mglRenderBindableLocValid(member_loc,
                                                   MAX_BINDABLE_BUFFERS)) {
                        continue;
                    }

                    if (sm->is_array_member) {
                        GLuint elem_stride = sm->member_array_stride;
                        GLuint src_stride = mglRenderStructPackSrcStride(
                            sm->member_src_stride, elem_stride);
                        for (GLint ai = 0; ai < (GLint)sm->member_size; ai++) {
                            GLint elem_loc = member_loc + ai;
                            if (!mglRenderBindableLocValid(elem_loc,
                                                           MAX_BINDABLE_BUFFERS)) {
                                continue;
                            }
                            BufferBaseTarget *mb = &buffers[elem_loc];
                            Buffer *mbuf = mglRendererGetValidatedBuffer(
                                ctx, mb->buf,
                                "mapShaderBufferResourcesViaPlan(struct,array)",
                                (unsigned long)elem_loc);
                            if (!mbuf && allowFallback) {
                                BufferBaseTarget *fb = &fallbackBuffers[elem_loc];
                                mbuf = mglRendererGetValidatedBuffer(
                                    ctx, fb->buf,
                                    "mapShaderBufferResourcesViaPlan(struct,array,fb)",
                                    (unsigned long)elem_loc);
                            }
                            if (!mglRenderCPUShadowReadable(
                                    mbuf ? mbuf->data.buffer_data : NULL,
                                    mbuf ? mbuf->size : 0)) {
                                continue;
                            }
                            if (mglRenderStructPackUseBulk(
                                    ai, mbuf->size, sm->member_size, src_stride)) {
                                if (elem_stride == src_stride) {
                                    size_t copy_size = (size_t)mglRenderClampCopyToStruct(
                                        (uint64_t)member_offset, (uint64_t)mbuf->size,
                                        (uint64_t)struct_size);
                                    if (copy_size > 0) {
                                        memcpy(packed + member_offset,
                                               (const void *)(uintptr_t)mbuf->data.buffer_data,
                                               copy_size);
                                    }
                                } else {
                                    const uint8_t *src =
                                        (const uint8_t *)(uintptr_t)
                                            mbuf->data.buffer_data;
                                    for (GLint sj = 0; sj < (GLint)sm->member_size;
                                         sj++) {
                                        size_t dest_off =
                                            (size_t)member_offset +
                                            (size_t)sj * (size_t)elem_stride;
                                        size_t copy_size =
                                            (size_t)mglRenderClampCopyToStruct(
                                                (uint64_t)dest_off,
                                                (uint64_t)src_stride,
                                                (uint64_t)struct_size);
                                        if (copy_size == 0)
                                            break;
                                        memcpy(packed + dest_off,
                                               src + (size_t)sj * src_stride,
                                               copy_size);
                                    }
                                }
                                break;
                            }
                            size_t dest_off = (size_t)member_offset +
                                (size_t)ai * (size_t)elem_stride;
                            size_t copy_size = (size_t)mbuf->size;
                            if (copy_size > (size_t)src_stride) {
                                copy_size = (size_t)src_stride;
                            }
                            copy_size = (size_t)mglRenderClampCopyToStruct(
                                (uint64_t)dest_off, (uint64_t)copy_size,
                                (uint64_t)struct_size);
                            if (copy_size > 0) {
                                memcpy(packed + dest_off,
                                       (const void *)(uintptr_t)mbuf->data.buffer_data,
                                       copy_size);
                            }
                        }
                    } else {
                        BufferBaseTarget *mb = &buffers[member_loc];
                        Buffer *mbuf = mglRendererGetValidatedBuffer(
                            ctx, mb->buf,
                            "mapShaderBufferResourcesViaPlan(struct,scalar)",
                            (unsigned long)member_loc);
                        if (!mbuf && allowFallback) {
                            BufferBaseTarget *fb = &fallbackBuffers[member_loc];
                            mbuf = mglRendererGetValidatedBuffer(
                                ctx, fb->buf,
                                "mapShaderBufferResourcesViaPlan(struct,scalar,fb)",
                                (unsigned long)member_loc);
                        }
                        if (!mglRenderCPUShadowReadable(
                                mbuf ? mbuf->data.buffer_data : NULL,
                                mbuf ? mbuf->size : 0)) {
                            continue;
                        }
                        size_t copy_size = (size_t)mglRenderClampCopyToStruct(
                            (uint64_t)member_offset,
                            mbuf ? (uint64_t)mbuf->size : 0u,
                            (uint64_t)struct_size);
                        if (copy_size > 0) {
                            memcpy(packed + member_offset,
                                   (const void *)(uintptr_t)mbuf->data.buffer_data,
                                   copy_size);
                        }
                    }
                }

                if (getenv("MGL_DEBUG_STRUCT_PACK")) {
                    const float *fv = (const float *)packed;
                    fprintf(stderr,
                            "MGL STRUCTDUMP prog=%u stage=%d res=%s elem=%u loc=%d metal=%u size=%lu\n",
                            (unsigned)program->name, stage,
                            resource->name ? resource->name : "(null)",
                            element, base_loc + (GLint)(loc_step * element),
                            (unsigned)metal_binding, (unsigned long)struct_size);
                    for (size_t di = 0; di < struct_size && di < 64; di += 4) {
                        fprintf(stderr,
                                "  off[%zu] = %02x%02x%02x%02x (float=%.6f)\n",
                                di, packed[di], packed[di+1], packed[di+2], packed[di+3],
                                fv[di/4]);
                    }
                }

                Buffer *packedBuf = mglGetPackedStructBuffer(packed, struct_size);
                if (packed != stack_packed) {
                    free(packed);
                }
                if (!packedBuf) {
                    continue;
                }

                if (!mglRenderMappedBufferCountOK(
                        (uint32_t)buffer_map->count, MAX_MAPPED_BUFFERS)) {
                    fprintf(stderr,
                            "MGL ERROR: mapShaderBufferResourcesViaPlan struct overflow: count=%d max=%d\n",
                            buffer_map->count, MAX_MAPPED_BUFFERS);
                    return false;
                }
                BufferMap *bentry = &buffer_map->buffers[buffer_map->count];
                bzero(bentry, sizeof(*bentry));
                bentry->attribute_mask = 0;
                bentry->buffer_base_index = (GLuint)(base_loc + (GLint)(loc_step * element));
                bentry->resource_type = (GLuint)spvc_type;
                bentry->resource_index = entry->resource_index;
                bentry->metal_binding_index = metal_binding;
                bentry->has_metal_binding = (GLboolean)mglRenderGLBoolean(1);
                bentry->buf = packedBuf;
                bentry->offset = 0;
                bentry->size = (GLsizeiptr)struct_size;
                buffer_map->count++;
            }
            continue;  /* next plan entry */
        }

        /* ---- Normal binding path ---- */
        for (GLuint element = 0; element < entry->element_count; element++) {
            GLuint metal_binding = mglBufferPlanMetalBindingForElement(entry, element);
            GLuint spirv_binding = mglBufferPlanClientBindingForElement(entry, resource, element);
            if (!mglRenderClientBindingInRange(spirv_binding,
                                               MAX_BINDABLE_BUFFERS)) {
                static uint64_t s_planOverflowHits = 0;
                uint64_t hit = ++s_planOverflowHits;
                if (hit <= 16ull || (hit % 4096ull) == 0ull) {
                    fprintf(stderr,
                            "MGL WARNING: mapShaderBufferResourcesViaPlan: stage=%d type=%d binding=%u exceeds MAX_BINDABLE_BUFFERS=%d, skipping (hit=%llu)\n",
                            stage, spvc_type, spirv_binding, MAX_BINDABLE_BUFFERS,
                            (unsigned long long)hit);
                }
                continue;
            }

            BufferBaseTarget *baseBinding = &buffers[spirv_binding];
            bool usedFallbackBinding = false;
            bool allowGlobalFallback = mglRenderAllowGlobalBufferFallback(
                fallbackBuffers ? 1 : 0, spvc_type, entry->flags) != 0;
            if (allowGlobalFallback &&
                mglRenderBufferBindingEmpty(baseBinding->buf ? 1 : 0,
                                            baseBinding->buffer)) {
                BufferBaseTarget *fallbackBinding = &fallbackBuffers[spirv_binding];
                if (fallbackBinding->buf || fallbackBinding->buffer != 0) {
                    baseBinding = fallbackBinding;
                    usedFallbackBinding = true;
                }
            }
            Buffer *buf = mglRendererGetValidatedBuffer(ctx, baseBinding->buf,
                                                        "mapShaderBufferResourcesViaPlan(base)",
                                                        (unsigned long)spirv_binding);

            /* Recover from name/object map skew. */
            if (!buf && baseBinding->buffer != 0) {
                Buffer *resolved = (Buffer *)searchHashTable(&state->buffer_table,
                                                             baseBinding->buffer);
                resolved = mglRendererGetValidatedBuffer(ctx, resolved,
                                                         "mapShaderBufferResourcesViaPlan(base,recover)",
                                                         (unsigned long)spirv_binding);
                if (resolved) {
                    baseBinding->buf = resolved;
                    buf = resolved;
                    static unsigned long long s_recoverHits = 0;
                    if ((++s_recoverHits % 64ull) == 1ull) {
                        fprintf(stderr,
                                "MGL BUFFER RECOVER: stage=%d type=%d binding=%u name=%u ptr=%p hit=%llu (plan)\n",
                                stage, spvc_type, spirv_binding, baseBinding->buffer, resolved,
                                s_recoverHits);
                    }
                }
            }

            size_t reflectedRequiredSize = entry->required_size;

            if (buf) {
                if (!mglRenderMappedBufferCountOK(
                        (uint32_t)buffer_map->count, MAX_MAPPED_BUFFERS)) {
                    fprintf(stderr,
                            "MGL ERROR: mapShaderBufferResourcesViaPlan overflow: count=%d max=%d\n",
                            buffer_map->count, MAX_MAPPED_BUFFERS);
                    return false;
                }
                BufferMap *bentry = &buffer_map->buffers[buffer_map->count];
                bzero(bentry, sizeof(*bentry));
                bentry->attribute_mask = 0;
                bentry->buffer_base_index = spirv_binding;
                bentry->resource_type = (GLuint)spvc_type;
                bentry->resource_index = entry->resource_index;
                bentry->metal_binding_index = metal_binding;
                bentry->has_metal_binding = (GLboolean)mglRenderGLBoolean(1);
                bentry->buf = buf;
                bentry->offset = baseBinding->offset;
                bentry->size = baseBinding->size;
                bentry->size = mglRenderMappedUniformSize(
                    spvc_type, bentry->size, buf->size, bentry->offset,
                    (uint64_t)reflectedRequiredSize);
                baseBinding->buffer = buf->name;
                buffer_map->count++;

                if (mglProgramNeedsBindingTrace(program)) {
                    static uint64_t s_focusedUBOMapLogs = 0;
                    if (mglShouldLogFocusedBinding(&s_focusedUBOMapLogs)) {
                        fprintf(stderr,
                                "MGL BINDMAP focused program=%u stage=%s type=%s resource=%s resourceIndex=%d clientBinding=%u metalSlot=%u buffer=%u offset=%lld range=%lld reflected=%lu (plan)\n",
                                (unsigned)program->name,
                                mglShaderStageName(stage),
                                mglMGLShaderResourceTypeName(spvc_type),
                                resource->name ? resource->name : "(null)",
                                entry->resource_index,
                                (unsigned)spirv_binding,
                                (unsigned)metal_binding,
                                (unsigned)buf->name,
                                (long long)baseBinding->offset,
                                (long long)baseBinding->size,
                                (unsigned long)reflectedRequiredSize);
                    }
                }

                static uint64_t s_traceFileUBOMapLogs = 0;
                if (mglProgramNeedsTraceLog(program) &&
                    mglShouldLogTraceFileBindingForProgram(program, &s_traceFileUBOMapLogs)) {
                    mglTraceLog("BINDMAP program=%u stage=%s type=%s resource=%s resourceIndex=%d clientBinding=%u metalSlot=%u buffer=%u offset=%lld range=%lld reflected=%lu fallback=%d (plan)",
                                (unsigned)program->name,
                                mglShaderStageName(stage),
                                mglMGLShaderResourceTypeName(spvc_type),
                                resource->name ? resource->name : "(null)",
                                entry->resource_index,
                                (unsigned)spirv_binding,
                                (unsigned)metal_binding,
                                (unsigned)buf->name,
                                (long long)baseBinding->offset,
                                (long long)baseBinding->size,
                                (unsigned long)reflectedRequiredSize,
                                usedFallbackBinding ? 1 : 0);
                }

                if (mglRenderBaseBindingTooSmall(baseBinding->size,
                                                 (uint64_t)reflectedRequiredSize)) {
                    GLuint programName = ctx ? state->program_name : 0u;
                    if (mglShouldLogSmallBaseBinding(programName,
                                                     stage,
                                                     spvc_type,
                                                     spirv_binding,
                                                     buf->name,
                                                     baseBinding->size,
                                                     reflectedRequiredSize)) {
                        fprintf(stderr,
                                "MGL WARNING: base binding too small program=%u stage=%d type=%d binding=%u glName=%u range=%lld reflected=%lu (padding at bind) (plan)\n",
                                programName,
                                stage,
                                spvc_type,
                                spirv_binding,
                                buf->name,
                                (long long)baseBinding->size,
                                (unsigned long)reflectedRequiredSize);
                    }
                }
            } else {
                if (mglProgramNeedsBindingTrace(program)) {
                    static uint64_t s_focusedUBOMissLogs = 0;
                    if (mglShouldLogFocusedBinding(&s_focusedUBOMissLogs)) {
                        fprintf(stderr,
                                "MGL BINDMISS focused program=%u stage=%s type=%s resource=%s resourceIndex=%d clientBinding=%u metalSlot=%u baseBuffer=%u basePtr=%p offset=%lld range=%lld reflected=%lu usedFallback=%d (plan)\n",
                                (unsigned)program->name,
                                mglShaderStageName(stage),
                                mglMGLShaderResourceTypeName(spvc_type),
                                resource->name ? resource->name : "(null)",
                                entry->resource_index,
                                (unsigned)spirv_binding,
                                (unsigned)metal_binding,
                                baseBinding->buffer,
                                baseBinding->buf,
                                (long long)baseBinding->offset,
                                (long long)baseBinding->size,
                                (unsigned long)reflectedRequiredSize,
                                usedFallbackBinding ? 1 : 0);
                    }
                }
                static uint64_t s_traceFileUBOMissLogs = 0;
                if (mglProgramNeedsTraceLog(program) &&
                    mglShouldLogTraceFileBindingForProgram(program, &s_traceFileUBOMissLogs)) {
                    mglTraceLog("BINDMISS program=%u stage=%s type=%s resource=%s resourceIndex=%d clientBinding=%u metalSlot=%u baseBuffer=%u basePtr=%p offset=%lld range=%lld reflected=%lu fallback=%d (plan)",
                                (unsigned)program->name,
                                mglShaderStageName(stage),
                                mglMGLShaderResourceTypeName(spvc_type),
                                resource->name ? resource->name : "(null)",
                                entry->resource_index,
                                (unsigned)spirv_binding,
                                (unsigned)metal_binding,
                                baseBinding->buffer,
                                baseBinding->buf,
                                (long long)baseBinding->offset,
                                (long long)baseBinding->size,
                                (unsigned long)reflectedRequiredSize,
                                usedFallbackBinding ? 1 : 0);
                }
                if (baseBinding->buf || baseBinding->buffer != 0 || baseBinding->offset != 0 || baseBinding->size != 0) {
                    static uint64_t s_dropInvalidHits = 0;
                    uint64_t hit = ++s_dropInvalidHits;
                    if (hit <= 16ull || (hit % 4096ull) == 0ull) {
                        fprintf(stderr,
                                "MGL WARNING: mapShaderBufferResourcesViaPlan: dropping invalid base buffer binding=%u stage=%d type=%d name=%u ptr=%p offset=%lld size=%lld (hit=%llu)\n",
                                spirv_binding, stage, spvc_type,
                                baseBinding->buffer,
                                baseBinding->buf,
                                (long long)baseBinding->offset,
                                (long long)baseBinding->size,
                                (unsigned long long)hit);
                    }
                    bzero(baseBinding, sizeof(BufferBaseTarget));
                }
                continue;
            }
        }
    }

    return true;
}

bool mglRendererMapShaderBufferResourcesToBufferMap(void *renderer,
                                                    BufferMapList *buffer_map,
                                                    int stage)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;

    /* The cached buffer binding plan is the only mapping path: it caches every
     * decision the previous reflection walk recomputed per draw (metal/client
     * binding bases, element counts, required sizes, skip rules, plain-uniform
     * struct packing metadata, global-fallback allowance) and is rebuilt at
     * link and after glUniformBlockBinding / glShaderStorageBlockBinding
     * mutations, so its per-draw output is what the reflection walk produced.
     *
     * Resolution order:
     *   1. no program for the stage -> nothing to map (the reflection walk also
     *      iterated zero resources here);
     *   2. cached plan valid        -> replay it;
     *   3. plan/stage invalid       -> force one rebuild (EnsureBuilt only
     *      rebuilds when the *vertex* stage is invalid) and replay;
     *   4. still invalid            -> allocation failure: refuse the draw
     *      loudly instead of binding a partial resource set.  The caller keeps
     *      the draw dirty and retries, so a transient failure recovers on the
     *      next frame. */
    Program *program = mglResolveProgramForStageFromState(ctx, stage);
    if (!program) {
        return true;
    }

    const MGLBufferBindingPlan *plan = mglBufferBindingPlanEnsureBuilt(program);
    const MGLStageBufferPlan *stagePlan = mglStageBufferPlan(plan, stage);
    if (!stagePlan || !stagePlan->valid) {
        mglBufferBindingPlanBuild(program);
        plan = mglBufferBindingPlanEnsureBuilt(program);
        stagePlan = mglStageBufferPlan(plan, stage);
    }
    if (!stagePlan || !stagePlan->valid) {
        static uint64_t s_planMissingHits = 0;
        uint64_t hit = ++s_planMissingHits;
        if (hit <= 16ull || (hit % 4096ull) == 0ull) {
            fprintf(stderr,
                    "MGL ERROR: buffer binding plan unavailable for program=%u stage=%d (plan=%p, hit=%llu); refusing the draw\n",
                    (unsigned)program->name, stage, (const void *)plan,
                    (unsigned long long)hit);
        }
        return false;
    }

    return mglRendererMapShaderBufferResourcesViaPlan(renderer, buffer_map, stage,
                                                      program, stagePlan);
}

bool mglRendererMapBuffersToMTL(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMState *state = mglBufferMapState(&areas);

    const int vertexStage = areas.tess_native_tes_active
        ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;
    if (mglRendererMapGLBuffersToMTLBufferMap(renderer,
            &state->vertex_buffer_map_list, vertexStage) == false) {
        return false;
    }

    if (mglRendererMapGLBuffersToMTLBufferMap(renderer,
            &state->fragment_buffer_map_list, _FRAGMENT_SHADER) == false) {
        return false;
    }

    return true;
}

/* === copy-on-write snapshot pool ======================================== */

uint64_t mglAdvanceFrameGeneration(void)
{
    return mglRenderAdvanceBufferGeneration();
}

void mglRecordFrameCompleted(uint64_t generation)
{
    mglRenderRecordBufferGenerationCompleted(generation);
}

/* Mark the slot holding buf's current Metal backing as encoded in the current
 * generation, so it is not recycled until that frame's GPU work completes. */
void mglNoteBufferEncoded(Buffer *buf)
{
    mglRenderNoteBufferEncoded(buf);
}

/* === dirty-buffer bookkeeping =========================================== */

bool mglRendererUpdateDirtyBuffer(void *renderer, Buffer *ptr)
{
    (void)renderer;
    char error[256] = {0};
    int result = mglRenderUpdateDirtyBuffer(ptr, error, sizeof(error));
    if (result == MGL_RENDER_BUFFER_OPERATION_HANDLED) {
        return true;
    }
    fprintf(stderr, "MGL BUFFER ERROR: Metal-cpp dirty update failed buffer=%u: %s\n",
            ptr ? ptr->name : 0u, error[0] ? error : "?");
    return false;
}

bool mglRendererCheckForDirtyBufferData(void *renderer,
                                        BufferMapList *buffer_map_list)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return mglRenderCheckForDirtyBufferData(areas.ctx, buffer_map_list,
                                            MGL_BUFFER_MAP_WHERE_CHECK_DIRTY);
}

bool mglRendererUpdateDirtyBaseBufferList(void *renderer,
                                          BufferMapList *buffer_map_list)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    return mglRenderUpdateDirtyBaseBufferList(areas.ctx, buffer_map_list,
                                              MGL_BUFFER_MAP_WHERE_UPDATE_DIRTY);
}

int mglRendererGetVertexBufferIndexWithAttributeSet(void *renderer,
                                                    int attribute)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    return mglRenderVertexBufferIndexForAttribute(ctx, mglBufferMapState(&areas),
                                                  attribute,
                                                  MGL_BUFFER_MAP_WHERE_VERTEX_INDEX);
}

/* MGL_RENDERER_RESOURCE_STORAGE_SHARED is 0 (the enum lives in the
 * Objective-C header; same twin other C hosts use). */
enum { MGL_PD_RESOURCE_STORAGE_SHARED = 0 };

/* Twins of MGLRenderer.m's static buffer helpers. */
static void *mglBmCreateBuffer(uint64_t length, uint64_t options)
{
    void *buffer = NULL;
    if (mglRenderCreateBuffer((size_t)length, options, NULL, &buffer) == 0 &&
        buffer) {
        return buffer;
    }
    return NULL;
}

static void *mglBmBufferContents(void *buffer)
{
    void *contents = NULL;
    uint64_t length = 0u;
    return buffer && mglRenderGetBufferContents(buffer, &contents, &length) == 0
               ? contents
               : NULL;
}

static uint64_t mglBmBufferLength(void *buffer)
{
    void *contents = NULL;
    uint64_t length = 0u;
    if (!buffer ||
        mglRenderGetBufferContents(buffer, &contents, &length) != 0) {
        return 0u;
    }
    return length;
}

/* -isolatedStageBindingBufferForMap:source:requiredLength: (log 201).
 * The port handed back a +1 handle (the method returned +0 and the caller took
 * its own reference), which mglBmCreateBuffer preserves. */
void *mglBufferIsolatedStageBinding(void *renderer, const BufferMap *map,
                                    void *source, uint64_t requiredLength)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    (void)areas;
    if (!map || !map->buf || requiredLength == 0) {
        return NULL;
    }

    void *isolated = mglBmCreateBuffer(requiredLength,
                                       MGL_PD_RESOURCE_STORAGE_SHARED);
    if (!isolated || !mglBmBufferContents(isolated)) {
        return NULL;
    }

    memset(mglBmBufferContents(isolated), 0, requiredLength);
    /* For UBOs, prefer the CPU shadow when present: the Metal backing may
     * not yet reflect a recent glBufferData before the first draw bind. */
    if (mglRenderIsolateUBOPrefersCPUShadow(
            (uint32_t)map->resource_type, map->buf != NULL,
            map->buf && map->buf->data.buffer_data, map->offset)) {
        size_t copyLength = mglBufferMapAvailableBackingBytes(
            map, (size_t)map->buf->size);
        copyLength = (size_t)mglRenderIsolateCopyLength(copyLength,
                                                        requiredLength);
        if (copyLength > 0) {
            memcpy(mglBmBufferContents(isolated),
                   ((const uint8_t *)(uintptr_t)map->buf->data.buffer_data) +
                       (size_t)map->offset,
                   copyLength);
            return isolated;
        }
    }

    if (!source || map->offset < 0 || !mglBmBufferContents(source)) {
        return isolated;
    }

    /* For UBOs, prefer the underlying store over the (possibly short) indexed
     * range so trailing std140 members remain visible after padding. */
    size_t copyLength = mglRenderIsolateUBOUsesFullStore(
                            (uint32_t)map->resource_type)
        ? mglBufferMapAvailableBackingBytes(map, mglBmBufferLength(source))
        : mglBufferMapVisibleBackingBytes(map, mglBmBufferLength(source));
    copyLength = (size_t)mglRenderIsolateCopyLength(copyLength, requiredLength);
    if (copyLength > 0) {
        memcpy(mglBmBufferContents(isolated),
               ((const uint8_t *)mglBmBufferContents(source)) + (size_t)map->offset,
               copyLength);
    }
    return isolated;
}
