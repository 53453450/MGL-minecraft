/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Extracted from MGLRenderer+Buffer.m.
 */

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Buffer_Private.h"
#import "mgl_buffer_plan.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"


static id mglBufferCreateConvertedVertexBuffer(
    Buffer *sourceBuffer,
    const MGLResolvedVertexAttribBinding *resolved,
    MGLRenderVertexConversionKind kind,
    GLuint componentCount,
    GLenum sourceType,
    GLboolean normalized,
    BOOL destinationSigned,
    NSUInteger *outStride)
{
    MGLRenderVertexConversion conversion = {0};
    conversion.kind = (uint32_t)kind;
    conversion.component_count = componentCount;
    conversion.source_type = sourceType;
    conversion.normalized = normalized ? 1u : 0u;
    conversion.destination_signed = destinationSigned ? 1u : 0u;
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
              (unsigned)kind,
              error[0] ? error : "?");
        return nil;
    }
    if (outStride) {
        *outStride = (NSUInteger)convertedStride;
    }
    return (__bridge_transfer id)convertedBuffer;
}

@implementation MGLRenderer (BufferBridge)

- (id)floatVertexBufferForDoubleAttrib:(Buffer *)sourceBuffer
                                         resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                             size:(GLuint)componentCount
                                         outStride:(NSUInteger *)outStride
{
    if (outStride) {
        *outStride = 0;
    }
    if (!sourceBuffer || !resolved || componentCount == 0 || componentCount > 4) {
        return nil;
    }
    return mglBufferCreateConvertedVertexBuffer(
        sourceBuffer, resolved, MGL_RENDER_VERTEX_DOUBLE_TO_FLOAT,
        componentCount, GL_DOUBLE, GL_FALSE, NO, outStride);
}

/* Metal has no int/uint->float vertex format conversion for 32-bit integer
 * formats (MGLVertexFormatInt/UInt require integer shader inputs). When an
 * app uses glVertexAttribFormat (non-integer) with GL_INT/GL_UNSIGNED_INT and
 * a float shader input, GL requires the integer values to be converted to
 * float. We perform that conversion on the CPU side, mirroring the GL_DOUBLE
 * path. sizeof(GLint)==sizeof(GLfloat)==4, so the converted stride equals the
 * original stride. */
- (id)floatVertexBufferForIntAttrib:(Buffer *)sourceBuffer
                                      resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                          size:(GLuint)componentCount
                                    normalized:(GLboolean)normalized
                                          type:(GLenum)type
                                     outStride:(NSUInteger *)outStride
{
    if (outStride) {
        *outStride = 0;
    }
    if (!sourceBuffer || !resolved || componentCount == 0 || componentCount > 4 ||
        (type != GL_INT && type != GL_UNSIGNED_INT)) {
        return nil;
    }
    return mglBufferCreateConvertedVertexBuffer(
        sourceBuffer, resolved, MGL_RENDER_VERTEX_INT_TO_FLOAT,
        componentCount, type, normalized, NO, outStride);
}

/* GL_FIXED: each component is a 32-bit signed integer (GLfixed) representing
 * a 16.16 fixed-point value (actual value = raw / 65536.0). size ranges 1-4;
 * each component is converted independently to float. Output is float[size].
 * sizeof(GLfixed)==sizeof(GLfloat)==4, so the converted stride equals the
 * original stride, mirroring floatVertexBufferForIntAttrib. */
- (id)floatVertexBufferForFixedAttrib:(Buffer *)sourceBuffer
                                         resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                             size:(GLuint)componentCount
                                        outStride:(NSUInteger *)outStride
{
    if (outStride) {
        *outStride = 0;
    }
    if (!sourceBuffer || !resolved || componentCount == 0 || componentCount > 4) {
        return nil;
    }
    return mglBufferCreateConvertedVertexBuffer(
        sourceBuffer, resolved, MGL_RENDER_VERTEX_FIXED_TO_FLOAT,
        componentCount, GL_FIXED, GL_FALSE, NO, outStride);
}

/* GL_UNSIGNED_INT_10_10_10_2: 1 uint32 packed as RGBA.
 * Non-REV bit layout: R[22-31] G[12-21] B[2-11] A[0-1].
 * Converted to float4(R/1023.0, G/1023.0, B/1023.0, A/3.0). The source
 * element (4 bytes) is smaller than the float4 output (16 bytes), so the
 * converted buffer is zero-initialized and the unpacked floats are written
 * per vertex (no copy-then-overwrite, unlike the GL_DOUBLE path). */
- (id)floatVertexBufferForPacked1010102Attrib:(Buffer *)sourceBuffer
                                                  resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                                 outStride:(NSUInteger *)outStride
{
    if (outStride) {
        *outStride = 0;
    }
    if (!sourceBuffer || !resolved) {
        return nil;
    }
    return mglBufferCreateConvertedVertexBuffer(
        sourceBuffer, resolved, MGL_RENDER_VERTEX_PACKED_1010102_TO_FLOAT,
        4u, GL_UNSIGNED_INT_10_10_10_2, GL_TRUE, NO, outStride);
}

/* GL_UNSIGNED_INT_10F_11F_11F_REV: 1 uint32 packed as RGB float.
 * REV bit layout: R[0-10] G[11-21] B[22-31].
 * R/G are 11-bit float, B is 10-bit float (all unsigned). Converted to
 * float3. Like the 10_10_10_2 path, the source element (4 bytes) is smaller
 * than the float3 output (12 bytes), so the converted buffer is zero-
 * initialized and unpacked floats are written per vertex. */
- (id)floatVertexBufferForPacked10f11f11fAttrib:(Buffer *)sourceBuffer
                                                     resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                                    outStride:(NSUInteger *)outStride
{
    if (outStride) {
        *outStride = 0;
    }
    if (!sourceBuffer || !resolved) {
        return nil;
    }
    return mglBufferCreateConvertedVertexBuffer(
        sourceBuffer, resolved, MGL_RENDER_VERTEX_PACKED_10F11F11F_TO_FLOAT,
        3u, GL_UNSIGNED_INT_10F_11F_11F_REV, GL_FALSE, NO, outStride);
}

/* Converts integer vertex data from a source type that Metal cannot feed
 * directly to an int/uint shader input (e.g. GL_UNSIGNED_BYTE -> int32 for
 * an `in int` attribute) into a 32-bit integer buffer matching the shader's
 * declared type. dstIsInt selects int32 vs uint32 output. */
- (id)integerVertexBufferForAttrib:(Buffer *)sourceBuffer
                                     resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                         size:(GLuint)componentCount
                                       srcType:(GLenum)srcType
                                     dstIsInt:(BOOL)dstIsInt
                                    outStride:(NSUInteger *)outStride
{
    if (outStride) {
        *outStride = 0;
    }
    if (!sourceBuffer || !resolved || componentCount == 0 || componentCount > 4) {
        return nil;
    }
    return mglBufferCreateConvertedVertexBuffer(
        sourceBuffer, resolved, MGL_RENDER_VERTEX_INTEGER_TO_32,
        componentCount, srcType, GL_FALSE, dstIsInt, outStride);
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

    int count;
    int mapped_buffers;
    struct {
        int spvc_type;
        int gl_buffer_type;
        const char *name;
    } mapped_types[4] = {
        {_UNIFORM_BUFFER_RES, _UNIFORM_BUFFER, "Uniform Buffer"},
        {_UNIFORM_CONSTANT_RES, _UNIFORM_CONSTANT, "Uniform Constant"},
        {_STORAGE_BUFFER_RES, _SHADER_STORAGE_BUFFER, "Shader Storage Buffer"},
        {_ATOMIC_COUNTER_RES, _ATOMIC_COUNTER_BUFFER, "Atomic Counter Buffer"}
    };
#if DEBUG_MAPPED_TYPES
    const char *stages[] = {"VERTEX_SHADER", "TESS_CONTROL_SHADER", "TESS_EVALUATION_SHADER",
        "GEOMETRY_SHADER", "FRAGMENT_SHADER", "COMPUTE_SHADER"};
#endif
    
    // init mapped buffer count
    buffer_map->count = 0;

    if (![self mapShaderBufferResourcesToBufferMap:buffer_map stage:stage]) {
        return false;
    }

    // bind vao attribs to buffers (attribs can share the same buffer)
    if (stage == _VERTEX_SHADER)
    {
        int count = mglRendererGetProgramBindingCount(ctx, stage, _STAGE_INPUT_RES);
        VertexArray *vao = mglRendererGetValidatedVAO(ctx, "mapGLBuffersToMTLBufferMap");
        if (![self mapVertexAttributeBuffersToBufferMap:buffer_map vao:vao stageInputCount:count stage:stage]) {
            return false;
        }
    }
    else if (stage == _COMPUTE_SHADER)
    {
    }

    if (kMGLDiagnosticStateLogs && mglShouldTraceCall(mapCall)) {
        mglTraceLogNSString(@"MGL TRACE map.end stage=%d call=%llu mappedCount=%u",
              stage,
              (unsigned long long)mapCall,
              buffer_map ? buffer_map->count : 0);
    }

    return true;
}

/* Fast path: map shader buffer resources using the cached buffer binding plan.
 * Returns true if the plan was valid and all resources were processed.
 * Returns false if the plan is unavailable (caller falls back to the
 * reflection-based path in mapShaderBufferResourcesToBufferMap). */
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

        if (entry->flags & MGL_BP_FLAG_SKIP) {
            continue;
        }

        /* Validate the resource is still in range (plan was built from the
         * same list, but guard against any unexpected reallocation). */
        if (spvc_type < 0 || spvc_type >= MGL_MAX_SHADER_RESOURCES ||
            entry->resource_index >= program->shader_resources_list[stage][spvc_type].count) {
            return false;  /* fall back to original path */
        }
        MGLShaderResource *resource =
            &program->shader_resources_list[stage][spvc_type].list[entry->resource_index];

        /* Resolve buffer arrays (same logic as the original path). */
        int gl_buffer_type = -1;
        switch (spvc_type) {
            case _UNIFORM_BUFFER_RES:     gl_buffer_type = _UNIFORM_BUFFER; break;
            case _UNIFORM_CONSTANT_RES:   gl_buffer_type = _UNIFORM_CONSTANT; break;
            case _STORAGE_BUFFER_RES:    gl_buffer_type = _SHADER_STORAGE_BUFFER; break;
            case _ATOMIC_COUNTER_RES:    gl_buffer_type = _ATOMIC_COUNTER_BUFFER; break;
            default: return false;
        }

        BufferBaseTarget *buffers;
        BufferBaseTarget *fallbackBuffers = NULL;
        if (spvc_type == _UNIFORM_CONSTANT_RES) {
            buffers = program->plain_uniform_buffers;
            fallbackBuffers = MGL_STATE(ctx)->buffer_base[gl_buffer_type].buffers;
        } else {
            buffers = MGL_STATE(ctx)->buffer_base[gl_buffer_type].buffers;
        }

        /* MGL_DEBUG_STRUCT_PACK diagnostic (gated by getenv). */
        if (spvc_type == _UNIFORM_CONSTANT_RES &&
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
        if (entry->flags & MGL_BP_FLAG_STRUCT_PACKED)
        {
            GLuint loc_step = entry->loc_step;
            GLint base_loc = entry->base_loc;
            GLuint struct_size = entry->struct_size;
            GLuint array_size = entry->element_count;
            bool allowFallback = fallbackBuffers &&
                (entry->flags & MGL_BP_FLAG_ALLOW_FALLBACK);

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
                    if (member_loc_off < elem_loc_start ||
                        member_loc_off >= elem_loc_end) {
                        continue;
                    }

                    GLuint member_offset = sm->member_offset_in_elem;
                    if (member_offset >= elem_byte_start) {
                        member_offset -= elem_byte_start;
                    }
                    if (member_offset >= struct_size) {
                        continue;
                    }

                    GLint member_loc = sm->member_loc;
                    if (member_loc < 0 || member_loc >= (GLint)MAX_BINDABLE_BUFFERS) {
                        continue;
                    }

                    if (sm->is_array_member) {
                        GLuint elem_stride = sm->member_array_stride;
                        GLuint src_stride = sm->member_src_stride
                                                ? sm->member_src_stride
                                                : elem_stride;
                        for (GLint ai = 0; ai < (GLint)sm->member_size; ai++) {
                            GLint elem_loc = member_loc + ai;
                            if (elem_loc < 0 || elem_loc >= (GLint)MAX_BINDABLE_BUFFERS) {
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
                            if (!mbuf || !mbuf->data.buffer_data || mbuf->size <= 0) {
                                continue;
                            }
                            if (ai == 0 &&
                                (size_t)mbuf->size >=
                                    (size_t)sm->member_size * src_stride) {
                                if (elem_stride == src_stride) {
                                    size_t copy_size = (size_t)mbuf->size;
                                    if ((size_t)member_offset + copy_size > struct_size)
                                        copy_size = struct_size - (size_t)member_offset;
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
                                        if (dest_off >= struct_size)
                                            break;
                                        size_t copy_size = src_stride;
                                        if (dest_off + copy_size > struct_size)
                                            copy_size =
                                                struct_size - dest_off;
                                        if (copy_size > 0)
                                            memcpy(packed + dest_off,
                                                   src + (size_t)sj * src_stride,
                                                   copy_size);
                                    }
                                }
                                break;
                            }
                            size_t copy_size = (size_t)mbuf->size;
                            if (copy_size > (size_t)src_stride) {
                                copy_size = (size_t)src_stride;
                            }
                            size_t dest_off = (size_t)member_offset +
                                (size_t)ai * (size_t)elem_stride;
                            if (dest_off >= struct_size) {
                                continue;
                            }
                            if (dest_off + copy_size > struct_size) {
                                copy_size = struct_size - dest_off;
                            }
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
                        if (!mbuf || !mbuf->data.buffer_data || mbuf->size <= 0) {
                            continue;
                        }
                        size_t copy_size = (size_t)mbuf->size;
                        if ((size_t)member_offset + copy_size > struct_size) {
                            copy_size = struct_size - (size_t)member_offset;
                        }
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

                if (buffer_map->count >= MAX_MAPPED_BUFFERS) {
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
                bentry->has_metal_binding = GL_TRUE;
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
            GLuint client_binding = mglBufferPlanClientBindingForElement(entry, resource, element);
            if (client_binding >= MAX_BINDABLE_BUFFERS) {
                static uint64_t s_planOverflowHits = 0;
                uint64_t hit = ++s_planOverflowHits;
                if (hit <= 16ull || (hit % 4096ull) == 0ull) {
                    NSLog(@"MGL WARNING: mapShaderBufferResourcesViaPlan: stage=%d type=%d binding=%u exceeds MAX_BINDABLE_BUFFERS=%d, skipping (hit=%llu)",
                          stage, spvc_type, client_binding, MAX_BINDABLE_BUFFERS,
                          (unsigned long long)hit);
                }
                continue;
            }

            BufferBaseTarget *baseBinding = &buffers[client_binding];
            bool usedFallbackBinding = false;
            bool allowGlobalFallback =
                fallbackBuffers &&
                (spvc_type != _UNIFORM_CONSTANT_RES ||
                 (entry->flags & MGL_BP_FLAG_ALLOW_FALLBACK));
            if (allowGlobalFallback && !baseBinding->buf && baseBinding->buffer == 0) {
                BufferBaseTarget *fallbackBinding = &fallbackBuffers[client_binding];
                if (fallbackBinding->buf || fallbackBinding->buffer != 0) {
                    baseBinding = fallbackBinding;
                    usedFallbackBinding = true;
                }
            }
            Buffer *buf = mglRendererGetValidatedBuffer(ctx, baseBinding->buf,
                                                        "mapShaderBufferResourcesViaPlan(base)",
                                                        (NSUInteger)client_binding);

            /* Recover from name/object map skew. */
            if (!buf && baseBinding->buffer != 0) {
                Buffer *resolved = (Buffer *)searchHashTable(&MGL_STATE(ctx)->buffer_table, baseBinding->buffer);
                resolved = mglRendererGetValidatedBuffer(ctx, resolved,
                                                         "mapShaderBufferResourcesViaPlan(base,recover)",
                                                         (NSUInteger)client_binding);
                if (resolved) {
                    baseBinding->buf = resolved;
                    buf = resolved;
                    static unsigned long long s_recoverHits = 0;
                    if ((++s_recoverHits % 64ull) == 1ull) {
                        NSLog(@"MGL BUFFER RECOVER: stage=%d type=%d binding=%u name=%u ptr=%p hit=%llu (plan)",
                              stage, spvc_type, client_binding, baseBinding->buffer, resolved,
                              s_recoverHits);
                    }
                }
            }

            NSUInteger reflectedRequiredSize = entry->required_size;

            if (buf) {
                if (buffer_map->count >= MAX_MAPPED_BUFFERS) {
                    NSLog(@"MGL ERROR: mapShaderBufferResourcesViaPlan overflow: count=%d max=%d",
                          buffer_map->count, MAX_MAPPED_BUFFERS);
                    return false;
                }
                BufferMap *bentry = &buffer_map->buffers[buffer_map->count];
                bzero(bentry, sizeof(*bentry));
                bentry->attribute_mask = 0;
                bentry->buffer_base_index = client_binding;
                bentry->resource_type = (GLuint)spvc_type;
                bentry->resource_index = entry->resource_index;
                bentry->metal_binding_index = metal_binding;
                bentry->has_metal_binding = GL_TRUE;
                bentry->buf = buf;
                bentry->offset = baseBinding->offset;
                bentry->size = baseBinding->size;
                if (spvc_type == _UNIFORM_BUFFER_RES) {
                    bentry->size = mglBufferMapExtendUniformRange(
                        bentry->size, buf->size, bentry->offset,
                        reflectedRequiredSize);
                }
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
                              (unsigned)client_binding,
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
                                (unsigned)client_binding,
                                (unsigned)metal_binding,
                                (unsigned)buf->name,
                                (long long)baseBinding->offset,
                                (long long)baseBinding->size,
                                (unsigned long)reflectedRequiredSize,
                                usedFallbackBinding ? 1 : 0);
                }

                if (reflectedRequiredSize > 0 && baseBinding->size > 0 &&
                    (NSUInteger)baseBinding->size < reflectedRequiredSize) {
                    GLuint programName = ctx ? MGL_STATE(ctx)->program_name : 0u;
                    if (mglShouldLogSmallBaseBinding(programName,
                                                     stage,
                                                     spvc_type,
                                                     client_binding,
                                                     buf->name,
                                                     baseBinding->size,
                                                     reflectedRequiredSize)) {
                        NSLog(@"MGL WARNING: base binding too small program=%u stage=%d type=%d binding=%u glName=%u range=%lld reflected=%lu (padding at bind) (plan)",
                              programName,
                              stage,
                              spvc_type,
                              client_binding,
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
                              (unsigned)client_binding,
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
                                (unsigned)client_binding,
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
                              client_binding, stage, spvc_type,
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
    /* Resolve the active program once and reuse it across both the fast path
     * and the reflection fallback, avoiding repeated per-resource re-resolves. */
    Program *program = mglResolveProgramForStageFromState(ctx, stage);

    /* Fast path: try the cached buffer binding plan first.  If the plan
     * is valid, this skips all per-draw name lookups, program resolution,
     * and MSL argument scans for resources that haven't changed since
     * link.  Falls through to the original reflection-based path if the
     * plan is unavailable (NULL program, plan not yet built, or stage
     * invalid after a binding mutation). */
    if (program) {
        const MGLBufferBindingPlan *plan =
            mglBufferBindingPlanEnsureBuilt(program);
        const MGLStageBufferPlan *stagePlan = mglStageBufferPlan(plan, stage);
        if (stagePlan && stagePlan->valid) {
            return [self mapShaderBufferResourcesViaPlan:buffer_map
                                                    stage:stage
                                                  program:program
                                                stagePlan:stagePlan];
        }
    }

    /* Original reflection-based path (plan unavailable or invalid). */
    int count;
    struct {
        int spvc_type;
        int gl_buffer_type;
        const char *name;
    } mapped_types[4] = {
        {_UNIFORM_BUFFER_RES, _UNIFORM_BUFFER, "Uniform Buffer"},
        {_UNIFORM_CONSTANT_RES, _UNIFORM_CONSTANT, "Uniform Constant"},
        {_STORAGE_BUFFER_RES, _SHADER_STORAGE_BUFFER, "Shader Storage Buffer"},
        {_ATOMIC_COUNTER_RES, _ATOMIC_COUNTER_BUFFER, "Atomic Counter Buffer"}
    };
#if DEBUG_MAPPED_TYPES
    const char *stages[] = {"VERTEX_SHADER", "TESS_CONTROL_SHADER", "TESS_EVALUATION_SHADER",
        "GEOMETRY_SHADER", "FRAGMENT_SHADER", "COMPUTE_SHADER"};
#endif

    for(int type=0; type<4; type++)
    {
        int spvc_type;
        int gl_buffer_type;

        spvc_type = mapped_types[type].spvc_type;
        gl_buffer_type = mapped_types[type].gl_buffer_type;

        /* Read count directly from the already-resolved program instead of
         * getProgramBindingCount:, which would re-resolve the program. */
        count = (program && spvc_type >= 0 && spvc_type < MGL_MAX_SHADER_RESOURCES)
                  ? (int)program->shader_resources_list[stage][spvc_type].count
                  : 0;

#if DEBUG_MAPPED_TYPES
        DEBUG_PRINT("Checking mapped_types: %s count:%d for stage: %s\n", mapped_types[type].name, count, stages[stage]);
#endif

        if (count)
        {
            BufferBaseTarget *buffers;
            BufferBaseTarget *fallbackBuffers = NULL;

            if (spvc_type == _UNIFORM_CONSTANT_RES && program) {
                buffers = program->plain_uniform_buffers;
                fallbackBuffers = MGL_STATE(ctx)->buffer_base[gl_buffer_type].buffers;
            } else {
                buffers = MGL_STATE(ctx)->buffer_base[gl_buffer_type].buffers;
            }
            
            for (int i = 0; i < count; i++)
            {
                GLuint client_binding;
                Buffer *buf;
                BufferBaseTarget *baseBinding;

                // Use the GL binding point to locate the client's buffer base.
                // The resource's `binding` may already have been rewritten to the
                // Metal [[buffer(n)]] slot parsed from generated MSL.
                if (!program || spvc_type < 0 || spvc_type >= MGL_MAX_SHADER_RESOURCES ||
                    i >= (int)program->shader_resources_list[stage][spvc_type].count) {
                    continue;
                }
                MGLShaderResource *resource = &program->shader_resources_list[stage][spvc_type].list[i];
                if (mglShouldSkipStageBufferResource(program, stage, spvc_type, resource)) {
                    continue;
                }

                if (spvc_type == _UNIFORM_CONSTANT_RES &&
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

                /* Plain struct uniform packing.
                 *
                 * The AIR backend translates `layout(location=N) uniform S u[K]`
                 * into separate Metal buffer arguments (`constant S* u_0
                 * [[buffer(B)]]`, etc.), each expecting a full struct's
                 * worth of data.  MGL stores individual uniform member data
                 * per location in plain_uniform_buffers[location].  Pack
                 * the member data into struct-sized Metal buffers here. */
                if (spvc_type == _UNIFORM_CONSTANT_RES &&
                    resource->ubo_members && resource->ubo_member_count > 0 &&
                    resource->required_size > 0 &&
                    !mglRendererResourceLooksSamplerLike(resource, spvc_type)) {

                    GLuint loc_step = mglPlainStructLocStep(resource);
                    GLint base_loc = resource->uniform_location;
                    if (base_loc < 0) {
                        base_loc = (GLint)resource->location;
                    }
                    GLuint array_size = mglStageBufferResourceElementCount(spvc_type, resource);
                    size_t struct_size = resource->required_size;
                    bool allowFallback = fallbackBuffers &&
                        mglPlainUniformAllowsGlobalFallback(resource);

                    for (GLuint element = 0; element < array_size; element++) {
                        GLuint metal_binding = mglMetalResourceSlotForElement(resource, element);
                        GLuint elem_loc_start = element * loc_step;
                        GLuint elem_loc_end = (element + 1u) * loc_step;
                        GLuint elem_byte_start = element * (GLuint)struct_size;

                        uint8_t stack_packed[256];
                        uint8_t *packed = (struct_size <= sizeof(stack_packed))
                                          ? stack_packed
                                          : (uint8_t *)calloc(1, struct_size);
                        if (!packed) continue;
                        memset(packed, 0, struct_size);

                        for (GLuint m = 0; m < resource->ubo_member_count; m++) {
                            SpirvUBOMember *member = &resource->ubo_members[m];

                            /* member->location_offset is relative to the
                             * resource's base uniform_location (spans all
                             * array elements).  Filter to current element. */
                            GLuint member_loc_off = (GLuint)member->location_offset;
                            if (member_loc_off < elem_loc_start ||
                                member_loc_off >= elem_loc_end) {
                                continue;
                            }

                            /* member->offset is the absolute byte offset
                             * across the whole array.  Compute the relative
                             * offset within this element. */
                            GLuint member_offset = member->offset;
                            if (member_offset >= elem_byte_start) {
                                member_offset -= elem_byte_start;
                            }
                            if (member_offset >= struct_size) {
                                continue;
                            }

                            /* Location of this member's data in
                             * plain_uniform_buffers: base_loc + the member's
                             * absolute location_offset. */
                            GLint member_loc = base_loc + (GLint)member_loc_off;
                            if (member_loc < 0 || member_loc >= (GLint)MAX_BINDABLE_BUFFERS) {
                                continue;
                            }

                            if (member->size > 1) {
                                /* Array member: each element at its own
                                 * location (CTS).  Nested struct paths use
                                 * std140 ArrayStride; top-level plain arrays
                                 * stay tightly packed for LLVM loads. */
                                GLuint src_stride =
                                    mglGLTypeElementByteSize(member->gl_type);
                                GLuint elem_stride = src_stride;
                                if (member->name && strchr(member->name, '.') &&
                                    member->array_stride > (GLint)src_stride &&
                                    member->array_stride > 0) {
                                    elem_stride = (GLuint)member->array_stride;
                                } else if (elem_stride == 0) {
                                    elem_stride = (GLuint)member->array_stride;
                                    src_stride = elem_stride ? elem_stride : 4u;
                                }
                                if (src_stride == 0)
                                    src_stride = 4u;
                                for (GLint ai = 0; ai < member->size; ai++) {
                                    GLint elem_loc = member_loc + ai;
                                    if (elem_loc < 0 || elem_loc >= (GLint)MAX_BINDABLE_BUFFERS) {
                                        continue;
                                    }
                                    BufferBaseTarget *mb = &buffers[elem_loc];
                                    Buffer *mbuf = mglRendererGetValidatedBuffer(
                                        ctx, mb->buf,
                                        "mapGLBuffersToMTLBufferMap(struct,array)",
                                        (NSUInteger)elem_loc);
                                    if (!mbuf && allowFallback) {
                                        BufferBaseTarget *fb = &fallbackBuffers[elem_loc];
                                        mbuf = mglRendererGetValidatedBuffer(
                                            ctx, fb->buf,
                                            "mapGLBuffersToMTLBufferMap(struct,array,fb)",
                                            (NSUInteger)elem_loc);
                                    }
                                    if (!mbuf || !mbuf->data.buffer_data || mbuf->size <= 0) {
                                        continue;
                                    }
                                    /* glUniform*iv/fv uploads an entire array
                                     * to the base location in one buffer.  Copy
                                     * once when source and dest strides match;
                                     * otherwise scatter leaf elements into the
                                     * std140 layout. */
                                    if (ai == 0 &&
                                        (size_t)mbuf->size >=
                                            (size_t)member->size * src_stride) {
                                        if (elem_stride == src_stride) {
                                            size_t copy_size = (size_t)mbuf->size;
                                            if ((size_t)member_offset + copy_size >
                                                struct_size)
                                                copy_size =
                                                    struct_size -
                                                    (size_t)member_offset;
                                            if (copy_size > 0) {
                                                memcpy(packed + member_offset,
                                                       (const void *)(uintptr_t)
                                                           mbuf->data.buffer_data,
                                                       copy_size);
                                            }
                                        } else {
                                            const uint8_t *src =
                                                (const uint8_t *)(uintptr_t)
                                                    mbuf->data.buffer_data;
                                            for (GLint sj = 0; sj < member->size;
                                                 sj++) {
                                                size_t dest_off =
                                                    (size_t)member_offset +
                                                    (size_t)sj *
                                                        (size_t)elem_stride;
                                                if (dest_off >= struct_size)
                                                    break;
                                                size_t copy_size = src_stride;
                                                if (dest_off + copy_size >
                                                    struct_size)
                                                    copy_size =
                                                        struct_size -
                                                        dest_off;
                                                if (copy_size > 0)
                                                    memcpy(packed + dest_off,
                                                           src +
                                                               (size_t)sj *
                                                                   src_stride,
                                                           copy_size);
                                            }
                                        }
                                        break;
                                    }
                                    size_t copy_size = (size_t)mbuf->size;
                                    if (copy_size > (size_t)elem_stride) {
                                        copy_size = (size_t)elem_stride;
                                    }
                                    size_t dest_off = (size_t)member_offset +
                                        (size_t)ai * (size_t)elem_stride;
                                    if (dest_off >= struct_size) {
                                        continue;
                                    }
                                    if (dest_off + copy_size > struct_size) {
                                        copy_size = struct_size - dest_off;
                                    }
                                    if (copy_size > 0) {
                                        memcpy(packed + dest_off,
                                               (const void *)(uintptr_t)mbuf->data.buffer_data,
                                               copy_size);
                                    }
                                }
                            } else {
                                /* Scalar / vector / matrix member: all data
                                 * at one location. */
                                BufferBaseTarget *mb = &buffers[member_loc];
                                Buffer *mbuf = mglRendererGetValidatedBuffer(
                                    ctx, mb->buf,
                                    "mapGLBuffersToMTLBufferMap(struct,scalar)",
                                    (NSUInteger)member_loc);
                                if (!mbuf && allowFallback) {
                                    BufferBaseTarget *fb = &fallbackBuffers[member_loc];
                                    mbuf = mglRendererGetValidatedBuffer(
                                        ctx, fb->buf,
                                        "mapGLBuffersToMTLBufferMap(struct,scalar,fb)",
                                        (NSUInteger)member_loc);
                                }
                                if (!mbuf || !mbuf->data.buffer_data || mbuf->size <= 0) {
                                    continue;
                                }
                                size_t copy_size = (size_t)mbuf->size;
                                if ((size_t)member_offset + copy_size > struct_size) {
                                    copy_size = struct_size - (size_t)member_offset;
                                }
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

                        if (buffer_map->count >= MAX_MAPPED_BUFFERS) {
                            NSLog(@"MGL ERROR: mapGLBuffersToMTLBufferMap struct overflow: count=%d max=%d",
                                  buffer_map->count, MAX_MAPPED_BUFFERS);
                            return false;
                        }
                        BufferMap *entry = &buffer_map->buffers[buffer_map->count];
                        bzero(entry, sizeof(*entry));
                        entry->attribute_mask = 0;
                        entry->buffer_base_index = (GLuint)(base_loc + (GLint)(loc_step * element));
                        entry->resource_type = (GLuint)spvc_type;
                        entry->resource_index = (GLuint)i;
                        entry->metal_binding_index = metal_binding;
                        entry->has_metal_binding = GL_TRUE;
                        entry->buf = packedBuf;
                        entry->offset = 0;
                        entry->size = (GLsizeiptr)struct_size;
                        buffer_map->count++;
                    }
                    continue; /* Skip normal binding path for struct resource */
                }

                GLuint element_count = mglStageBufferResourceElementCount(spvc_type, resource);
                for (GLuint element = 0; element < element_count; element++) {
                    GLuint metal_binding = mglMetalResourceSlotForElement(resource, element);
                    client_binding = mglClientBufferBindingForResourceElement(spvc_type, resource, element);
                    if (client_binding >= MAX_BINDABLE_BUFFERS)
                    {
                        NSLog(@"MGL WARNING: mapGLBuffersToMTLBufferMap: stage=%d type=%d binding=%u exceeds MAX_BINDABLE_BUFFERS=%d, skipping",
                              stage, spvc_type, client_binding, MAX_BINDABLE_BUFFERS);
                        continue;
                    }

                baseBinding = &buffers[client_binding];
                bool usedFallbackBinding = false;
                bool allowGlobalFallback =
                    fallbackBuffers &&
                    (spvc_type != _UNIFORM_CONSTANT_RES ||
                     mglPlainUniformAllowsGlobalFallback(resource));
                if (allowGlobalFallback && !baseBinding->buf && baseBinding->buffer == 0) {
                    BufferBaseTarget *fallbackBinding = &fallbackBuffers[client_binding];
                    if (fallbackBinding->buf || fallbackBinding->buffer != 0) {
                        baseBinding = fallbackBinding;
                        usedFallbackBinding = true;
                    }
                }
                buf = mglRendererGetValidatedBuffer(ctx, baseBinding->buf,
                                                    "mapGLBuffersToMTLBufferMap(base)",
                                                    (NSUInteger)client_binding);

                // Recover from name/object map skew: some paths can preserve GL name while pointer slot is stale.
                if (!buf && baseBinding->buffer != 0) {
                    Buffer *resolved = (Buffer *)searchHashTable(&MGL_STATE(ctx)->buffer_table, baseBinding->buffer);
                    resolved = mglRendererGetValidatedBuffer(ctx, resolved,
                                                             "mapGLBuffersToMTLBufferMap(base,recover)",
                                                             (NSUInteger)client_binding);
                    if (resolved) {
                        baseBinding->buf = resolved;
                        buf = resolved;
                        static unsigned long long s_recoverHits = 0;
                        if ((++s_recoverHits % 64ull) == 1ull) {
                            NSLog(@"MGL BUFFER RECOVER: stage=%d type=%d binding=%u name=%u ptr=%p hit=%llu",
	                              stage, spvc_type, client_binding, baseBinding->buffer, resolved,
	                              s_recoverHits);
                        }
	                    }
	                }

                /* Read required_size directly from the already-resolved
                 * resource instead of getProgramBindingRequiredSize:, which
                 * would re-resolve the program. */
                NSUInteger reflectedRequiredSize = (NSUInteger)resource->required_size;

	                if (buf)
	                {
	                    if (buffer_map->count >= MAX_MAPPED_BUFFERS)
	                    {
	                        NSLog(@"MGL ERROR: mapGLBuffersToMTLBufferMap overflow: count=%d max=%d",
                              buffer_map->count, MAX_MAPPED_BUFFERS);
                        return false;
                    }
                    BufferMap *entry = &buffer_map->buffers[buffer_map->count];
                    bzero(entry, sizeof(*entry));
                    entry->attribute_mask = 0; // non attribute.. no bits set
                    entry->buffer_base_index = client_binding;
                    entry->resource_type = (GLuint)spvc_type;
                    entry->resource_index = (GLuint)i;
                    entry->metal_binding_index = metal_binding;
                    entry->has_metal_binding = GL_TRUE;
                    entry->buf = buf;
                    entry->offset = baseBinding->offset;
                    entry->size = baseBinding->size;
                    if (spvc_type == _UNIFORM_BUFFER_RES) {
                        entry->size = mglBufferMapExtendUniformRange(
                            entry->size, buf->size, entry->offset,
                            reflectedRequiredSize);
                    }
                    baseBinding->buffer = buf->name;
                    buffer_map->count++;

                    if (mglProgramNeedsBindingTrace(program)) {
                        static uint64_t s_focusedUBOMapLogs = 0;
                        if (mglShouldLogFocusedBinding(&s_focusedUBOMapLogs)) {
                            NSLog(@"MGL BINDMAP focused program=%u stage=%s type=%s resource=%s resourceIndex=%d clientBinding=%u metalSlot=%u buffer=%u offset=%lld range=%lld reflected=%lu",
                                  (unsigned)program->name,
                                  mglShaderStageName(stage),
                                  mglMGLShaderResourceTypeName(spvc_type),
                                  resource->name ? resource->name : "(null)",
                                  i,
                                  (unsigned)client_binding,
                                  (unsigned)metal_binding,
                                  (unsigned)buf->name,
                                  (long long)baseBinding->offset,
                                  (long long)baseBinding->size,
                                  (unsigned long)reflectedRequiredSize);
                        }
                    }

                    /* Trace file: log UBO binding for program trace */
                    static uint64_t s_traceFileUBOMapLogs = 0;
                    if (mglProgramNeedsTraceLog(program) &&
                        mglShouldLogTraceFileBindingForProgram(program, &s_traceFileUBOMapLogs)) {
                        mglTraceLog("BINDMAP program=%u stage=%s type=%s resource=%s resourceIndex=%d clientBinding=%u metalSlot=%u buffer=%u offset=%lld range=%lld reflected=%lu fallback=%d",
                                    (unsigned)program->name,
                                    mglShaderStageName(stage),
                                    mglMGLShaderResourceTypeName(spvc_type),
                                    resource->name ? resource->name : "(null)",
                                    i,
                                    (unsigned)client_binding,
                                    (unsigned)metal_binding,
                                    (unsigned)buf->name,
                                    (long long)baseBinding->offset,
                                    (long long)baseBinding->size,
                                    (unsigned long)reflectedRequiredSize,
                                    usedFallbackBinding ? 1 : 0);
                    }

                    if (reflectedRequiredSize > 0 && baseBinding->size > 0 &&
                        (NSUInteger)baseBinding->size < reflectedRequiredSize) {
                        GLuint programName = ctx ? MGL_STATE(ctx)->program_name : 0u;
                        if (mglShouldLogSmallBaseBinding(programName,
                                                         stage,
                                                         spvc_type,
                                                         client_binding,
                                                         buf->name,
                                                         baseBinding->size,
                                                         reflectedRequiredSize)) {
                            NSLog(@"MGL WARNING: base binding too small program=%u stage=%d type=%d binding=%u glName=%u range=%lld reflected=%lu (padding at bind)",
                                  programName,
                                  stage,
                                  spvc_type,
                                  client_binding,
                                  buf->name,
                                  (long long)baseBinding->size,
                                  (unsigned long)reflectedRequiredSize);
                        }
                    }
                    
                    //DEBUG_PRINT("Found buffer type: %s buffer_base_index: %d\n", mapped_types[type].name, client_binding);
	                }
	                else
	                {
                    if (mglProgramNeedsBindingTrace(program)) {
                        static uint64_t s_focusedUBOMissLogs = 0;
                        if (mglShouldLogFocusedBinding(&s_focusedUBOMissLogs)) {
                            NSLog(@"MGL BINDMISS focused program=%u stage=%s type=%s resource=%s resourceIndex=%d clientBinding=%u metalSlot=%u baseBuffer=%u basePtr=%p offset=%lld range=%lld reflected=%lu usedFallback=%d",
                                  (unsigned)program->name,
                                  mglShaderStageName(stage),
                                  mglMGLShaderResourceTypeName(spvc_type),
                                  resource->name ? resource->name : "(null)",
                                  i,
                                  (unsigned)client_binding,
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
                        mglTraceLog("BINDMISS program=%u stage=%s type=%s resource=%s resourceIndex=%d clientBinding=%u metalSlot=%u baseBuffer=%u basePtr=%p offset=%lld range=%lld reflected=%lu fallback=%d",
                                    (unsigned)program->name,
                                    mglShaderStageName(stage),
                                    mglMGLShaderResourceTypeName(spvc_type),
                                    resource->name ? resource->name : "(null)",
                                    i,
                                    (unsigned)client_binding,
                                    (unsigned)metal_binding,
                                    (unsigned)baseBinding->buffer,
                                    baseBinding->buf,
                                    (long long)baseBinding->offset,
                                    (long long)baseBinding->size,
                                    (unsigned long)reflectedRequiredSize,
                                    usedFallbackBinding ? 1 : 0);
                    }
	                    if (baseBinding->buf || baseBinding->buffer != 0 || baseBinding->offset != 0 || baseBinding->size != 0) {
	                        NSLog(@"MGL WARNING: mapGLBuffersToMTLBufferMap: dropping invalid base buffer binding=%u stage=%d type=%d name=%u ptr=%p offset=%lld size=%lld",
	                              client_binding, stage, spvc_type,
                              baseBinding->buffer,
                              baseBinding->buf,
                              (long long)baseBinding->offset,
                              (long long)baseBinding->size);
                        bzero(baseBinding, sizeof(BufferBaseTarget));
                    }
                    // Some vanilla shader paths tolerate unbound blocks on specific stages.
                    // Skip instead of poisoning global GL error state with GL_INVALID_OPERATION.
                    continue;
                }
                }
            }
        }
    }

    return true;
}

- (bool)mapVertexAttributeBuffersToBufferMap:(BufferMapList *)buffer_map
                                         vao:(VertexArray *)vao
                            stageInputCount:(int)count
                                       stage:(int)stage
{
    int vao_buffer_start;
    int mapped_buffers = 0;
    GLuint next_vertex_binding_index = (GLuint)kMGLVertexAttribBufferBase;
    Program *activeProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);

    mapped_buffers = 0;

    if (!vao) {
        if (count > 0) {
            NSLog(@"MGL WARNING: mapGLBuffersToMTLBufferMap: stage inputs=%d but VAO is invalid/null, skipping attrib mapping",
                  count);
        }
        return true;
    }

    if (kMGLVertexAttribBufferBase >= kMGLMaxMetalVertexBufferCount) {
        NSLog(@"MGL ERROR: invalid vertex attrib base index=%lu (max valid=%lu)",
              (unsigned long)kMGLVertexAttribBufferBase,
              (unsigned long)kMGLMaxMetalVertexBufferIndex);
        return false;
    }

    // vao buffers start after the uniforms and shader buffers
    vao_buffer_start = buffer_map->count;
    // CRITICAL SECURITY FIX: Check against actual map capacity.
    if (buffer_map->count >= MAX_MAPPED_BUFFERS) {
        NSLog(@"MGL SECURITY ERROR: buffer_map count %d exceeds MAX_MAPPED_BUFFERS %d",
              buffer_map->count, MAX_MAPPED_BUFFERS);
        return false;
    }
    buffer_map->buffers[vao_buffer_start].attribute_mask = 0;
    buffer_map->buffers[vao_buffer_start].buffer_base_index = (GLuint)kMGLVertexAttribBufferBase;
    buffer_map->buffers[vao_buffer_start].resource_type = 0;
    buffer_map->buffers[vao_buffer_start].resource_index = 0;
    buffer_map->buffers[vao_buffer_start].metal_binding_index = 0;
    buffer_map->buffers[vao_buffer_start].has_metal_binding = GL_FALSE;
    buffer_map->buffers[vao_buffer_start].buf = NULL;
    buffer_map->buffers[vao_buffer_start].offset = 0;
    buffer_map->buffers[vao_buffer_start].size = 0;

    // create attribute map
    //
    // we need to cache this mapping, its called on each draw command
    //
    bool vaoHasExplicitAttribs = (vao->enabled_attribs != 0u);
    for(int att=0;att<MAX_ATTRIBS; att++)
    {
        if (vaoHasExplicitAttribs && !(vao->enabled_attribs & (0x1 << att)))
        {
            if ((vao->enabled_attribs >> (att+1)) == 0)
                break;
            continue;
        }
        {
            if (!mglRendererProgramUsesVertexAttrib(activeProgram, (GLuint)att)) {
                if (vaoHasExplicitAttribs && (vao->enabled_attribs >> (att+1)) == 0)
                    break;
                continue;
            }

            MGLResolvedVertexAttribBinding resolved = {0};
            if (!mglRendererResolveVertexAttribBinding(ctx,
                                                       vao,
                                                       (GLuint)att,
                                                       "mapGLBuffersToMTLBufferMap",
                                                       &resolved)) {
                NSLog(@"MGL WARNING: mapGLBuffersToMTLBufferMap: enabled attrib %d has invalid/NULL buffer, skipping attrib",
                      att);
                continue;
            }
            Buffer *gl_buffer = resolved.buffer;

            Buffer *map_buffer = NULL;

            // check start for map... then check
            map_buffer = buffer_map->buffers[vao_buffer_start].buf;

            // empty slot map it here, only works on first buffer..
            if (map_buffer == NULL)
            {
                if (next_vertex_binding_index >= kMGLMaxMetalVertexBufferCount) {
                    NSLog(@"MGL WARNING: vertex binding index overflow (next=%u maxValid=%lu), skipping attrib %d",
                          next_vertex_binding_index, (unsigned long)kMGLMaxMetalVertexBufferIndex, att);
                    continue;
                }
                // map the buffer object to a metal vertex index
                if (buffer_map->count >= MAX_MAPPED_BUFFERS) {
                    NSLog(@"MGL WARNING: vertex buffer map is full (count=%u max=%u), skipping attrib %d",
                          buffer_map->count, MAX_MAPPED_BUFFERS, att);
                    continue;
                }
                buffer_map->buffers[vao_buffer_start].attribute_mask |= (0x1 << att);
                buffer_map->buffers[vao_buffer_start].buf = gl_buffer;
                buffer_map->buffers[vao_buffer_start].buffer_base_index = next_vertex_binding_index++;
                buffer_map->buffers[vao_buffer_start].has_metal_binding = GL_FALSE;
                buffer_map->buffers[vao_buffer_start].offset = resolved.binding_offset;
                buffer_map->buffers[vao_buffer_start].size = 0;
                buffer_map->count++;

                mapped_buffers++;
            }
            else
            {
                bool found_buffer = false;

                // find vao attrib with same buffer
                for (int map=vao_buffer_start;
                     (found_buffer == false) && map<buffer_map->count;
                     map++)
                {
                    map_buffer = buffer_map->buffers[map].buf;
                    if (!map_buffer) {
                        continue;
                    }

                    // we need to check name and target, not pointers..
                    // FIX ME: I think we don't need a target as all attribs should be an array_buffer
                    // Offset is intentionally NOT compared: attributes sharing the same
                    // VBO/stride/divisor are grouped into one Metal buffer slot, with
                    // per-attribute offsets expressed via the vertex descriptor.
	                        if ((map_buffer->name == gl_buffer->name) &&
	                            (map_buffer->target == gl_buffer->target))
	                        {
	                            bool compatibleStream = true;
	                            for (GLuint prevAttrib = 0; prevAttrib < MAX_ATTRIBS; prevAttrib++) {
	                                if ((buffer_map->buffers[map].attribute_mask & (0x1u << prevAttrib)) == 0u) {
	                                    continue;
	                                }
	                                MGLResolvedVertexAttribBinding prevResolved = {0};
	                                if (!mglRendererResolveVertexAttribBinding(ctx,
	                                                                           vao,
	                                                                           prevAttrib,
	                                                                           "mapGLBuffersToMTLBufferMap(stream)",
	                                                                           &prevResolved)) {
	                                    continue;
	                                }
	                                if (prevResolved.stride != resolved.stride ||
	                                    prevResolved.divisor != resolved.divisor) {
	                                    compatibleStream = false;
	                                    break;
	                                }
	                            }
	                            if (compatibleStream) {
	                                // include it the list of attributes
	                                buffer_map->buffers[map].attribute_mask |= (0x1 << att);
	                                found_buffer = true;
	                                mapped_buffers++;
	                                break;
	                            }
	                        }
                }

                if (found_buffer == false)
                {
                    if (next_vertex_binding_index >= kMGLMaxMetalVertexBufferCount) {
                        NSLog(@"MGL WARNING: vertex binding index overflow (next=%u maxValid=%lu), cannot append attrib %d",
                              next_vertex_binding_index, (unsigned long)kMGLMaxMetalVertexBufferIndex, att);
                        continue;
                    }
                    // map the next buffer object to a metal vertex index
                    if (buffer_map->count >= MAX_MAPPED_BUFFERS) {
                        NSLog(@"MGL WARNING: vertex buffer map is full (count=%u max=%u), cannot append attrib %d",
                              buffer_map->count, MAX_MAPPED_BUFFERS, att);
                        continue;
                    }
                    buffer_map->buffers[buffer_map->count].attribute_mask = (0x1 << att);
                    buffer_map->buffers[buffer_map->count].buffer_base_index = next_vertex_binding_index++;
                    buffer_map->buffers[buffer_map->count].resource_type = 0;
                    buffer_map->buffers[buffer_map->count].resource_index = 0;
                    buffer_map->buffers[buffer_map->count].metal_binding_index = 0;
                    buffer_map->buffers[buffer_map->count].has_metal_binding = GL_FALSE;
                    buffer_map->buffers[buffer_map->count].buf = gl_buffer;
                    buffer_map->buffers[buffer_map->count].offset = resolved.binding_offset;
                    buffer_map->buffers[buffer_map->count].size = 0;
                    buffer_map->count++;

                    mapped_buffers++;
                }
            }
        }

        if (vaoHasExplicitAttribs && (vao->enabled_attribs >> (att+1)) == 0)
            break;
    }

    if (mapped_buffers != count) {
        static unsigned long long s_map_mismatch_hits = 0;
        s_map_mismatch_hits++;
        if ((s_map_mismatch_hits % 64ull) == 1ull) {
            Buffer *drawIndexBuffer = vao->element_array.buffer;
            void *indexBufferMetal = drawIndexBuffer ? drawIndexBuffer->data.mtl_data : NULL;
            NSLog(@"MGL WARNING: mapGLBuffersToMTLBufferMap mismatch (pipeline=%p mapped=%u expected=%u stage=%d hit=%llu indexBuffer=%p vao=%p)",
                  _pipelineCacheState.pipelineState, mapped_buffers, count, stage, s_map_mismatch_hits, indexBufferMetal, vao);
        }
    }

    return true;
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
    GLuint mapCount;

    if (!buffer_map_list) {
        return false;
    }

    mapCount = buffer_map_list->count;
    if (mapCount > MAX_MAPPED_BUFFERS) {
        NSLog(@"MGL WARNING: checkForDirtyBufferData mapCount=%u exceeds MAX_MAPPED_BUFFERS=%d, clamping",
              mapCount, MAX_MAPPED_BUFFERS);
        mapCount = MAX_MAPPED_BUFFERS;
    }

    // update vbos, some vbos may not have metal buffers yet
    for (GLuint i = 0; i < mapCount; i++)
    {
        Buffer *gl_buffer = mglRendererGetValidatedBuffer(ctx,
                                                          buffer_map_list->buffers[i].buf,
                                                          __FUNCTION__,
                                                          (NSUInteger)i);

        if (gl_buffer)
        {
            if (gl_buffer->data.dirty_bits)
            {
                return true;
            }
        } else if (buffer_map_list->buffers[i].buf) {
            buffer_map_list->buffers[i].buf = NULL;
        }
    }

    return false;
}

- (bool) updateDirtyBaseBufferList: (BufferMapList *)buffer_map_list
{
    GLuint mapCount;

    if (!buffer_map_list) {
        return true;
    }

    mapCount = buffer_map_list->count;
    if (mapCount > MAX_MAPPED_BUFFERS) {
        NSLog(@"MGL WARNING: updateDirtyBaseBufferList mapCount=%u exceeds MAX_MAPPED_BUFFERS=%d, clamping",
              mapCount, MAX_MAPPED_BUFFERS);
        mapCount = MAX_MAPPED_BUFFERS;
    }

    // update vbos, some vbos may not have metal buffers yet
    for (GLuint i = 0; i < mapCount; i++)
    {
        Buffer *gl_buffer = mglRendererGetValidatedBuffer(ctx,
                                                          buffer_map_list->buffers[i].buf,
                                                          __FUNCTION__,
                                                          (NSUInteger)i);

        if (gl_buffer)
        {
            if (gl_buffer->data.dirty_bits)
            {
                RETURN_FALSE_ON_FAILURE([self updateDirtyBuffer: gl_buffer]);
            }
        } else if (buffer_map_list->buffers[i].buf) {
            buffer_map_list->buffers[i].buf = NULL;
        }
    }

    return true;
}

/* bindVertexBuffersToCurrentRenderEncoder moved to MGLRenderer+Draw.m */

/* bindFragmentBuffersToCurrentRenderEncoder moved to MGLRenderer+Draw.m */

- (int) getVertexBufferIndexWithAttributeSet: (int) attribute
{
    if (attribute < 0 || attribute >= MAX_ATTRIBS) {
        NSLog(@"MGL ERROR: getVertexBufferIndexWithAttributeSet invalid attribute=%d", attribute);
        return -1;
    }

    VertexArray *vao = mglRendererGetValidatedVAO(ctx, __FUNCTION__);
    if (vao) {
        int resolved = mglRendererResolveVertexAttributeBufferIndex(ctx, vao, (GLuint)attribute, __FUNCTION__);
        if (resolved >= 0) {
            return resolved;
        }
    }

    // Legacy fallback: use cached map list if available.
    GLuint mapCount = MGL_STATE(ctx)->vertex_buffer_map_list.count;
    if (mapCount > MAX_MAPPED_BUFFERS) {
        mapCount = MAX_MAPPED_BUFFERS;
    }

    for (GLuint i = 0; i < mapCount; i++)
    {
        if (MGL_STATE(ctx)->vertex_buffer_map_list.buffers[i].attribute_mask & (0x1 << attribute)) {
            GLuint baseIndex = MGL_STATE(ctx)->vertex_buffer_map_list.buffers[i].buffer_base_index;
            if (baseIndex >= kMGLMaxMetalVertexBufferCount) {
                NSLog(@"MGL ERROR: getVertexBufferIndexWithAttributeSet mapped base index out of Metal range=%u (max valid=%lu)",
                      baseIndex, (unsigned long)kMGLMaxMetalVertexBufferIndex);
                return -1;
            }
            return (int)baseIndex;
        }
    }

    NSLog(@"MGL ERROR: No vertex buffer mapping found for attribute %d", attribute);
    return -1;
}


@end
