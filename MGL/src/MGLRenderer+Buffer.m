/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+Buffer.m
// Buffer/vertex data operations (GL buffer -> Metal mapping, dirty-buffer
// updates, vertex attribute conversion) extracted from MGLRenderer.m

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Buffer_Private.h"
#import "mgl_buffer_plan.h"
#import "mgl_vertex_attrib_plan.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"

static id mglBufferCreateConvertedVertexBuffer(
    Buffer *sourceBuffer,
    const MGLResolvedVertexAttribBinding *resolved,
    const MGLRenderVertexConversion *base,
    NSUInteger *outStride)
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
        NSLog(@"MGL BUFFER ERROR: Metal-cpp vertex conversion failed buffer=%u kind=%u: %s",
              sourceBuffer ? sourceBuffer->name : 0u,
              (unsigned)conversion.kind,
              error[0] ? error : "?");
        return nil;
    }
    if (outStride) {
        *outStride = (NSUInteger)convertedStride;
    }
    return (__bridge_transfer id)convertedBuffer;
}

@implementation MGLRenderer (Buffer)

- (id)convertedVertexBufferForAttribKind:(int)attribKind
                                  source:(Buffer *)sourceBuffer
                                resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                    size:(GLuint)componentCount
                                    type:(GLenum)type
                              normalized:(GLboolean)normalized
                               dstIsInt:(BOOL)dstIsInt
                               outStride:(NSUInteger *)outStride
{
    if (outStride) {
        *outStride = 0;
    }
    if (!sourceBuffer || !resolved) {
        return nil;
    }
    MGLRenderVertexConversion conversion = {0};
    if (mglRenderFillVertexConversionFromAttribKind(
            attribKind, (uint32_t)componentCount, (uint32_t)type,
            normalized ? 1 : 0, dstIsInt ? 1 : 0, &conversion) != 0) {
        return nil;
    }
    return mglBufferCreateConvertedVertexBuffer(sourceBuffer, resolved,
                                                &conversion, outStride);
}

/* bindMTLBuffer: moved to MGLRenderer+RenderPass.m */

/* bindMTLBufferLocked: moved to MGLRenderer+RenderPass.m */

/* ---- Plain struct uniform buffer packing ----
 *
 * The AIR backend translates `layout(location=N) uniform S u[K]` into separate
 * Metal buffer arguments (`constant S* u_0 [[buffer(B)]]`, etc.), each
 * expecting a full struct's worth of data.  MGL stores individual uniform
 * member data per location in plain_uniform_buffers[location].  This
 * packing logic combines individual member data into struct-sized Metal
 * buffers at render time.
 */

/* Compute the location step per array element from reflected members.
 * For a struct S = { vec4 m0, float m1[2], mat2 m2 }, the step is 4
 * (m0=1 + m1=2 + m2=1 in CTS convention). */
/* mglPlainStructLocStep and mglGLTypeElementByteSize are now shared
 * static inline helpers in mgl_buffer_plan.h. */

/* Resolver seam for the buffer-plan layer: the plan decides how resolved
 * attribute bindings group into Metal vertex buffer slots, while resolving one
 * attribute against live GL state stays here (it needs the context's validated
 * buffer table).  `where` is only used for validation diagnostics. */
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
        NSLog(@"MGL ERROR: Metal-cpp packed struct buffer failed: %s",
              error[0] ? error : "unknown");
    }
    return buffer;
}

- (bool) mapGLBuffersToMTLBufferMap:(BufferMapList *)buffer_map stage: (int) stage
{
    static uint64_t s_mapCallCountByStage[8] = {0};
    uint64_t mapCall = 0;
    if (stage >= 0 && stage < 8) {
        mapCall = ++s_mapCallCountByStage[stage];
    } else {
        mapCall = ++s_mapCallCountByStage[0];
    }

    if (kMGLDiagnosticStateLogs && mglShouldTraceCall(mapCall)) {
        mglTraceLogNSString(@"MGL TRACE map.begin stage=%d call=%llu preCount=%u program=%u",
              stage,
              (unsigned long long)mapCall,
              buffer_map ? buffer_map->count : 0,
              ctx ? (unsigned)MGL_STATE(ctx)->program_name : 0u);
    }

    // init mapped buffer count
    buffer_map->count = 0;

    if (![self mapShaderBufferResourcesToBufferMap:buffer_map stage:stage]) {
        return false;
    }

    // bind vao attribs to buffers (attribs can share the same buffer)
    if (mglRenderStageMapsVertexAttribs(stage))
    {
        const int count = mglRendererGetProgramBindingCount(ctx, stage, _STAGE_INPUT_RES);
        VertexArray *vao = mglRendererGetValidatedVAO(ctx, "mapGLBuffersToMTLBufferMap");
        if (!vao) {
            if (count > 0) {
                NSLog(@"MGL WARNING: mapGLBuffersToMTLBufferMap: stage inputs=%d but VAO is invalid/null, skipping attrib mapping",
                      count);
            }
        } else {
            /* Candidate attributes: enabled in the VAO (or every attribute when
             * the VAO carries no explicit enable mask) and consumed by the
             * program's vertex stage.  The grouping/limits themselves are the
             * buffer-plan layer's job (mglRenderPlanVertexAttribBuffers). */
            const BOOL explicitAttribMask = (vao->enabled_attribs != 0u);
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
                (const void *)_pipelineCache.state->pipelineState;
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

    if (kMGLDiagnosticStateLogs && mglShouldTraceCall(mapCall)) {
        mglTraceLogNSString(@"MGL TRACE map.end stage=%d call=%llu mappedCount=%u",
              stage,
              (unsigned long long)mapCall,
              buffer_map ? buffer_map->count : 0);
    }

    return true;
}

/* Map shader buffer resources from the cached buffer binding plan.  Called only
 * with a valid stage plan (see mapShaderBufferResourcesToBufferMap:stage:, which
 * owns plan availability); returns false when a resource cannot be bound. */
- (bool)mapShaderBufferResourcesViaPlan:(BufferMapList *)buffer_map
                                  stage:(int)stage
                                program:(Program *)program
                              stagePlan:(const MGLStageBufferPlan *)stagePlan
{
    if (!program || !stagePlan || !stagePlan->valid || !buffer_map) {
        return false;
    }

    for (uint32_t pi = 0; pi < stagePlan->entry_count; pi++)
    {
        const MGLBufferPlanEntry *entry = &stagePlan->entries[pi];
        int spvc_type = (int)entry->resource_type;

        if (mglRenderBufferPlanEntrySkip(entry->flags)) {
            continue;
        }

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
            fallbackBuffers = MGL_STATE(ctx)->buffer_base[gl_buffer_type].buffers;
        } else {
            buffers = MGL_STATE(ctx)->buffer_base[gl_buffer_type].buffers;
        }

        /* MGL_DEBUG_STRUCT_PACK diagnostic (gated by getenv). */
        if (mglRenderUsePlainUniformBuffers(spvc_type) &&
            getenv("MGL_DEBUG_STRUCT_PACK")) {
            NSLog(@"MGL STRUCTCHECK program=%u stage=%d name=%s ubo_members=%p count=%u req_size=%lu samplerLike=%d unifLoc=%d",
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
                                (NSUInteger)elem_loc);
                            if (!mbuf && allowFallback) {
                                BufferBaseTarget *fb = &fallbackBuffers[elem_loc];
                                mbuf = mglRendererGetValidatedBuffer(
                                    ctx, fb->buf,
                                    "mapShaderBufferResourcesViaPlan(struct,array,fb)",
                                    (NSUInteger)elem_loc);
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
                            (NSUInteger)member_loc);
                        if (!mbuf && allowFallback) {
                            BufferBaseTarget *fb = &fallbackBuffers[member_loc];
                            mbuf = mglRendererGetValidatedBuffer(
                                ctx, fb->buf,
                                "mapShaderBufferResourcesViaPlan(struct,scalar,fb)",
                                (NSUInteger)member_loc);
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
                    NSLog(@"MGL STRUCTDUMP prog=%u stage=%d res=%s elem=%u loc=%d metal=%u size=%lu",
                          (unsigned)program->name, stage,
                          resource->name ? resource->name : "(null)",
                          element, base_loc + (GLint)(loc_step * element),
                          (unsigned)metal_binding, (unsigned long)struct_size);
                    for (size_t di = 0; di < struct_size && di < 64; di += 4) {
                        NSLog(@"  off[%zu] = %02x%02x%02x%02x (float=%.6f)",
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
                    NSLog(@"MGL ERROR: mapShaderBufferResourcesViaPlan struct overflow: count=%d max=%d",
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
                    NSLog(@"MGL WARNING: mapShaderBufferResourcesViaPlan: stage=%d type=%d binding=%u exceeds MAX_BINDABLE_BUFFERS=%d, skipping (hit=%llu)",
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
                                                        (NSUInteger)spirv_binding);

            /* Recover from name/object map skew. */
            if (!buf && baseBinding->buffer != 0) {
                Buffer *resolved = (Buffer *)searchHashTable(&MGL_STATE(ctx)->buffer_table, baseBinding->buffer);
                resolved = mglRendererGetValidatedBuffer(ctx, resolved,
                                                         "mapShaderBufferResourcesViaPlan(base,recover)",
                                                         (NSUInteger)spirv_binding);
                if (resolved) {
                    baseBinding->buf = resolved;
                    buf = resolved;
                    static unsigned long long s_recoverHits = 0;
                    if ((++s_recoverHits % 64ull) == 1ull) {
                        NSLog(@"MGL BUFFER RECOVER: stage=%d type=%d binding=%u name=%u ptr=%p hit=%llu (plan)",
                              stage, spvc_type, spirv_binding, baseBinding->buffer, resolved,
                              s_recoverHits);
                    }
                }
            }

            NSUInteger reflectedRequiredSize = entry->required_size;

            if (buf) {
                if (!mglRenderMappedBufferCountOK(
                        (uint32_t)buffer_map->count, MAX_MAPPED_BUFFERS)) {
                    NSLog(@"MGL ERROR: mapShaderBufferResourcesViaPlan overflow: count=%d max=%d",
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
                        NSLog(@"MGL BINDMAP focused program=%u stage=%s type=%s resource=%s resourceIndex=%d clientBinding=%u metalSlot=%u buffer=%u offset=%lld range=%lld reflected=%lu (plan)",
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
                    GLuint programName = ctx ? MGL_STATE(ctx)->program_name : 0u;
                    if (mglShouldLogSmallBaseBinding(programName,
                                                     stage,
                                                     spvc_type,
                                                     spirv_binding,
                                                     buf->name,
                                                     baseBinding->size,
                                                     reflectedRequiredSize)) {
                        NSLog(@"MGL WARNING: base binding too small program=%u stage=%d type=%d binding=%u glName=%u range=%lld reflected=%lu (padding at bind) (plan)",
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
                        NSLog(@"MGL BINDMISS focused program=%u stage=%s type=%s resource=%s resourceIndex=%d clientBinding=%u metalSlot=%u baseBuffer=%u basePtr=%p offset=%lld range=%lld reflected=%lu usedFallback=%d (plan)",
                              (unsigned)program->name,
                              mglShaderStageName(stage),
                              mglMGLShaderResourceTypeName(spvc_type),
                              resource->name ? resource->name : "(null)",
                              entry->resource_index,
                              (unsigned)spirv_binding,
                              (unsigned)metal_binding,
                              (unsigned)baseBinding->buffer,
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
                                (unsigned)baseBinding->buffer,
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
                        NSLog(@"MGL WARNING: mapShaderBufferResourcesViaPlan: dropping invalid base buffer binding=%u stage=%d type=%d name=%u ptr=%p offset=%lld size=%lld (hit=%llu)",
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

- (bool)mapShaderBufferResourcesToBufferMap:(BufferMapList *)buffer_map stage:(int)stage
{
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
            NSLog(@"MGL ERROR: buffer binding plan unavailable for program=%u stage=%d (plan=%p, hit=%llu); refusing the draw",
                  (unsigned)program->name, stage, (const void *)plan,
                  (unsigned long long)hit);
        }
        return false;
    }

    return [self mapShaderBufferResourcesViaPlan:buffer_map
                                            stage:stage
                                          program:program
                                        stagePlan:stagePlan];
}


- (bool) mapBuffersToMTL
{
    const int vertexStage = _tessellation.nativeTESActive
        ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;
    if ([self mapGLBuffersToMTLBufferMap:
            &MGL_STATE(ctx)->vertex_buffer_map_list stage:vertexStage] == false)
        return false;

    if ([self mapGLBuffersToMTLBufferMap: &MGL_STATE(ctx)->fragment_buffer_map_list stage:_FRAGMENT_SHADER] == false)
        return false;

    return true;
}

/* Byte range the CPU shadow is allowed to push into the Metal store, clamped to
 * limit.  For a buffer a shader may have written (SSBO/atomic counter/transform
 * feedback) those writes are part of the data store per GL 4.6 §6.2 and live
 * only in the Metal buffer, so only the CPU-written range may be pushed and the
 * rest must be preserved.  The range is cumulative, so a CPU write covering the
 * whole store still authorizes a full overwrite.  Returns NO when there is
 * nothing to push. */
/* Buffer CoW generation and snapshot ownership live in the C++ backend. */

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

BOOL mglSnapshotSharedDirtyBuffer(Buffer *ptr, id *bufferPtr)
{
    void *metalBuffer = NULL;
    char error[256] = {0};
    if (mglRenderSnapshotSharedDirtyBuffer(
            ptr, &metalBuffer, error, sizeof(error)) != 0) {
        NSLog(@"MGL BUFFER ERROR: Metal-cpp dirty snapshot failed buffer=%u: %s",
              ptr ? ptr->name : 0u, error[0] ? error : "?");
        return NO;
    }
    if (bufferPtr) {
        *bufferPtr = (__bridge id)metalBuffer;
    }
    return YES;
}

BOOL mglSnapshotSharedBufferRange(Buffer *ptr,
                                  id *bufferPtr,
                                  NSUInteger offset,
                                  NSUInteger length)
{
    void *metalBuffer = NULL;
    char error[256] = {0};
    if (mglRenderSnapshotSharedBufferRange(
            ptr, offset, length, &metalBuffer, error, sizeof(error)) != 0) {
        NSLog(@"MGL BUFFER ERROR: Metal-cpp range snapshot failed buffer=%u: %s",
              ptr ? ptr->name : 0u, error[0] ? error : "?");
        return NO;
    }
    if (bufferPtr) {
        *bufferPtr = (__bridge id)metalBuffer;
    }
    return YES;
}

- (bool) updateDirtyBuffer:(Buffer *)ptr
{
    char error[256] = {0};
    int result = mglRenderUpdateDirtyBuffer(ptr, error, sizeof(error));
    if (result == MGL_RENDER_BUFFER_OPERATION_HANDLED) {
        return true;
    }
    NSLog(@"MGL BUFFER ERROR: Metal-cpp dirty update failed buffer=%u: %s",
          ptr ? ptr->name : 0u, error[0] ? error : "?");
    return false;
}

- (bool) checkForDirtyBufferData:  (BufferMapList *)buffer_map_list
{
    return mglRenderCheckForDirtyBufferData(ctx, buffer_map_list, __FUNCTION__);
}

- (bool) updateDirtyBaseBufferList: (BufferMapList *)buffer_map_list
{
    return mglRenderUpdateDirtyBaseBufferList(ctx, buffer_map_list, __FUNCTION__);
}

/* bindVertexBuffersToCurrentRenderEncoder moved to MGLRenderer+Draw.m */

/* bindFragmentBuffersToCurrentRenderEncoder moved to MGLRenderer+Draw.m */

- (int) getVertexBufferIndexWithAttributeSet: (int) attribute
{
    GLMState *state = MGL_STATE(ctx);
    return mglRenderVertexBufferIndexForAttribute(ctx, state, attribute, __FUNCTION__);
}

@end
