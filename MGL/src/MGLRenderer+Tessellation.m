// MGLRenderer+Tessellation.m
// Tessellation compute path (TCS/TES dispatch) extracted from MGLRenderer.m.
// GL_PATCHES draws run as consecutive Metal compute encoders: the TCS kernel
// writes per-patch output plus tess factors, then the TES kernel consumes them.

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Tessellation_Private.h"
#import "mgl_sampler_compat.h"
#import "mgl_trace_log.h"
#import "mgl_compute_pipeline_cache.h"
#include "mgl_env_flag.h"
#include "mgl_render_cpp_objc.h"
#include "mgl_shader_abi.h"
#include "mgl_air_gs_abi.h"
#include "mgl_air_tess_abi.h"

extern void mglRecordActivePrimitiveQueryDraw(GLMContext ctx, GLuint64 generated, GLuint64 written);

typedef enum MGLTCSStageInBaseType {
    MGLTCSStageInBaseFloat = 0,
    MGLTCSStageInBaseInt,
    MGLTCSStageInBaseUInt
} MGLTCSStageInBaseType;

typedef struct MGLTCSStageInMember {
    GLuint attribute;
    size_t offset;
    size_t size;
    size_t componentBytes;
    GLuint components;
    MGLTCSStageInBaseType baseType;
} MGLTCSStageInMember;

static void mglWriteTCSStageInComponent(
    uint8_t *destination,
    const MGLTCSStageInMember *member,
    size_t component,
    double value)
{
    if (!destination || !member || component >= member->components) {
        return;
    }

    uint8_t *component_destination = destination + member->offset +
        component * member->componentBytes;
    size_t copy_bytes = MIN(member->componentBytes, sizeof(int32_t));
    if (member->baseType == MGLTCSStageInBaseInt) {
        int32_t converted = (int32_t)value;
        memcpy(component_destination, &converted, copy_bytes);
    } else if (member->baseType == MGLTCSStageInBaseUInt) {
        uint32_t converted = value < 0.0 ? 0u : (uint32_t)value;
        memcpy(component_destination, &converted, copy_bytes);
    } else {
        float converted = (float)value;
        memcpy(component_destination, &converted, copy_bytes);
    }
}

static MGLMetalBufferRef mglTessCreateBuffer(MGLMetalDeviceRef device,
                                         NSUInteger length,
                                         MTLResourceOptions options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCppCreateBuffer(length, options, NULL, &buffer) == 0 &&
        buffer) {
        return (__bridge_transfer MGLMetalBufferRef)buffer;
    }
    return nil;
}

static MGLMetalBufferRef mglTessCreateBufferWithBytes(
    MGLMetalDeviceRef device,
    const void *bytes,
    NSUInteger length,
    MTLResourceOptions options)
{
    (void)device;
    void *buffer = NULL;
    if (mglRenderCppCreateBufferWithBytes(bytes, length, options, NULL,
                                          &buffer) == 0 && buffer) {
        return (__bridge_transfer MGLMetalBufferRef)buffer;
    }
    return nil;
}

static MGLMetalSamplerStateRef mglTessCreateSampler(
    MGLMetalDeviceRef device,
    MTLSamplerDescriptor *descriptor)
{
    (void)device;
    void *sampler = NULL;
    if (mglRenderCppCreateSampler((__bridge void *)descriptor,
                                  &sampler) == 0 && sampler) {
        return (__bridge_transfer MGLMetalSamplerStateRef)sampler;
    }
    return nil;
}

static MGLMetalTextureRef mglTessCreateTextureLevelView(
    MGLMetalTextureRef texture,
    NSUInteger level,
    NSUInteger sliceCount)
{
    void *view = NULL;
    if (mglRenderCppCreateTextureViewRange(
            (__bridge void *)texture, (uint32_t)texture.pixelFormat,
            (uint32_t)texture.textureType, level, 1, 0, sliceCount,
            0, 0, 0, 0, 0, &view) == 0 && view) {
        return (__bridge_transfer MGLMetalTextureRef)view;
    }
    return nil;
}

static void mglTessSetRenderVertexBuffer(MGLMetalRenderCommandEncoderRef encoder,
                                         void *renderEncoderOwner,
                                         MGLMetalBufferRef buffer,
                                         NSUInteger offset,
                                         NSUInteger index)
{
    (void)encoder;
    (void)mglRenderCppSetRenderBufferForOwner(
        renderEncoderOwner, (__bridge void *)buffer, offset,
        MGL_RENDER_CPP_BINDING_STAGE_VERTEX, (uint32_t)index);
}

static void mglTessDrawPrimitives(MGLMetalRenderCommandEncoderRef encoder,
                                  void *renderEncoderOwner,
                                  MTLPrimitiveType type,
                                  NSUInteger vertexStart,
                                  NSUInteger vertexCount,
                                  NSUInteger instanceCount,
                                  NSUInteger baseInstance)
{
    const MGLRenderCppDrawPlan plan = {
            .kind = MGL_RENDER_CPP_DRAW_ARRAY,
            .primitive_type = (uint32_t)type,
            .vertex_start = vertexStart,
            .vertex_count = vertexCount,
            .instance_count = instanceCount,
            .base_instance = baseInstance,
        };
    (void)encoder;
    (void)mglRenderCppEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

static bool mglTessEncodeBufferCopiesForOwner(
    void *commandBufferOwner,
    const MGLRenderCppBufferCopyEntry *entries,
    uint32_t entryCount)
{
    if (!commandBufferOwner || !entries || entryCount == 0u) return false;
    return mglRenderCppEncodeBufferCopiesForCommandBufferOwner(
        commandBufferOwner, entries, entryCount) == 0;
}

static bool mglTessAppendComputeResourceOp(
    MGLRenderCppComputeExecutionPlan *plan,
    NSMutableArray *temporaries,
    uint32_t kind,
    id resource,
    NSUInteger offset,
    NSUInteger index)
{
    if (!plan || kind > 3u ||
        plan->binding_op_count >= MGL_RENDER_CPP_COMPUTE_EXECUTION_MAX_OPS) {
        return false;
    }
    plan->binding_ops[plan->binding_op_count++] =
        (MGLRenderCppComputeBindingOp){
            .kind = kind,
            .index = (uint32_t)index,
            .offset = (uint64_t)offset,
            .buffer = (__bridge void *)resource,
            .bytes = NULL,
            .length = 0u,
        };
    if (resource && temporaries) [temporaries addObject:resource];
    return true;
}

static bool mglTessAppendComputeBytesOp(
    MGLRenderCppComputeExecutionPlan *plan,
    NSMutableArray *temporaries,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    if (!plan || !temporaries || !bytes || length == 0u ||
        length > UINT32_MAX ||
        plan->binding_op_count >= MGL_RENDER_CPP_COMPUTE_EXECUTION_MAX_OPS) {
        return false;
    }
    NSData *storage = [NSData dataWithBytes:bytes length:length];
    if (!storage) return false;
    [temporaries addObject:storage];
    plan->binding_ops[plan->binding_op_count++] =
        (MGLRenderCppComputeBindingOp){
            .kind = 1u,
            .index = (uint32_t)index,
            .offset = 0u,
            .buffer = NULL,
            .bytes = storage.bytes,
            .length = (uint32_t)length,
        };
    return true;
}

static bool mglTessPlanBufferOrBind(
    MGLRenderCppComputeExecutionPlan *plan,
    NSMutableArray *temporaries,
    MGLMetalComputeCommandEncoderRef encoder,
    MGLMetalBufferRef buffer,
    NSUInteger offset,
    NSUInteger index)
{
    (void)encoder;
    return mglTessAppendComputeResourceOp(
        plan, temporaries, 0u, buffer, offset, index);
}

static bool mglTessPlanTextureOrBind(
    MGLRenderCppComputeExecutionPlan *plan,
    NSMutableArray *temporaries,
    MGLMetalComputeCommandEncoderRef encoder,
    MGLMetalTextureRef texture,
    NSUInteger index)
{
    (void)encoder;
    return mglTessAppendComputeResourceOp(
        plan, temporaries, 2u, texture, 0u, index);
}

static bool mglTessPlanSamplerOrBind(
    MGLRenderCppComputeExecutionPlan *plan,
    NSMutableArray *temporaries,
    MGLMetalComputeCommandEncoderRef encoder,
    MGLMetalSamplerStateRef sampler,
    NSUInteger index)
{
    (void)encoder;
    return mglTessAppendComputeResourceOp(
        plan, temporaries, 3u, sampler, 0u, index);
}

static bool mglTessPlanBytesOrBind(
    MGLRenderCppComputeExecutionPlan *plan,
    NSMutableArray *temporaries,
    MGLMetalComputeCommandEncoderRef encoder,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    (void)encoder;
    return mglTessAppendComputeBytesOp(
        plan, temporaries, bytes, length, index);
}

static bool mglTessPlanDispatchOrBind(
    MGLRenderCppComputeExecutionPlan *plan,
    MGLMetalComputeCommandEncoderRef encoder,
    MTLSize groups,
    MTLSize threads)
{
    MGLRenderCppComputePlan dispatch = {
        .dispatch_kind = MGL_RENDER_CPP_COMPUTE_DISPATCH_DIRECT,
        .groups_x = (uint32_t)groups.width,
        .groups_y = (uint32_t)groups.height,
        .groups_z = (uint32_t)groups.depth,
        .local_x = (uint32_t)threads.width,
        .local_y = (uint32_t)threads.height,
        .local_z = (uint32_t)threads.depth,
        .indirect_buffer = NULL,
        .indirect_offset = 0u,
    };
    (void)encoder;
    return mglRenderCppAppendComputeDispatchToPlan(
        plan, &dispatch, NULL, 0) == 0;
}

static const uint8_t *mglRendererReadableBufferBytes(Buffer *buffer)
{
    if (!buffer) {
        return NULL;
    }
    if (buffer->data.buffer_data && ((uintptr_t)buffer->data.buffer_data >= 0x1000ull)) {
        return (const uint8_t *)(uintptr_t)buffer->data.buffer_data;
    }
    if (buffer->data.mtl_data) {
        MGLMetalBufferRef mtlBuffer = (__bridge MGLMetalBufferRef)(buffer->data.mtl_data);
        if (mtlBuffer && mtlBuffer.contents) {
            return (const uint8_t *)mtlBuffer.contents;
        }
    }
    return NULL;
}

@implementation MGLRenderer (Tessellation)

typedef struct {
    MGLMetalBufferRef __strong buffer;
    NSUInteger offset;
    MGLMetalBufferRef __strong initialization_source;
    NSUInteger initialization_source_offset;
    NSUInteger initialization_length;
    BOOL valid;
} MGLTessStageBufferBinding;

typedef struct {
    MGLTessStageBufferBinding slots[kMGLMaxBufferSlots];
    MGLMetalBufferRef __strong size_buffer;
    GLuint size_buffer_index;
} MGLTessStageBufferBindingList;

/* Tessellation shaders run as consecutive compute encoders. Prepare their
 * buffer bindings before opening the next encoder so an isolated binding can
 * be initialized by an ordered GPU copy from a buffer written by the previous
 * stage. Reading source.contents here would capture stale CPU bytes while the
 * preceding TCS encoder is still pending on the same command buffer. */
- (bool)prepareTessStageBufferBindings:(MGLTessStageBufferBindingList *)bindings
                                 stage:(int)stage
                             copyBacks:(MGLStageBindingCopyBackList *)copyBacks
{
    MGL_ASSERT_GL_THREAD();
    if (!bindings || !copyBacks) {
        return false;
    }

    BufferMapList stageBufferMap = {0};
    if (![self mapGLBuffersToMTLBufferMap:&stageBufferMap stage:stage]) {
        return false;
    }

    /* Complete every lazy allocation before creating the initialization blit
     * encoder. bindMTLBuffer: may itself need an encoder. */
    for (GLuint i = 0; i < stageBufferMap.count; i++) {
        Buffer *ptr = stageBufferMap.buffers[i].buf;
        if (ptr && !ptr->data.mtl_data) {
            [self bindMTLBuffer:ptr];
        }
    }

    for (GLuint i = 0; i < stageBufferMap.count; i++) {
        BufferMap *map = &stageBufferMap.buffers[i];
        Buffer *ptr = map->buf;
        if (!ptr) {
            continue;
        }

        NSUInteger metalBindingIndex = map->has_metal_binding
            ? (NSUInteger)map->metal_binding_index
            : (NSUInteger)map->buffer_base_index;
        if (metalBindingIndex >= kMGLMaxMetalVertexBufferCount) {
            continue;
        }
        [self clearStageBindingCopyBack:copyBacks atIndex:metalBindingIndex];
        if (map->offset < 0) {
            return false;
        }
        NSUInteger bindOffset = (NSUInteger)map->offset;

        MGLMetalBufferRef buffer = ptr->data.mtl_data
            ? (__bridge MGLMetalBufferRef)(ptr->data.mtl_data)
            : nil;
        if (buffer &&
            (ptr->data.dirty_bits & (DIRTY_BUFFER_DATA | DIRTY_BUFFER_ADDR))) {
            /* Consume the CPU-side initialization before a tessellation
             * stage can write the same Metal backing. Otherwise a later
             * stage bind would upload the stale shadow over the GPU result. */
            if (![self updateDirtyBuffer:ptr]) {
                return false;
            }
            buffer = ptr->data.mtl_data
                ? (__bridge MGLMetalBufferRef)(ptr->data.mtl_data)
                : nil;
        }
        NSUInteger requiredBytes =
            mglRendererGetProgramBindingRequiredSize(ctx, stage, (int)map->resource_type, (int)map->resource_index);
        if (map->resource_type == _ATOMIC_COUNTER_RES &&
            requiredBytes < sizeof(uint32_t)) {
            requiredBytes = sizeof(uint32_t);
        }

        GLsizeiptr storageRemaining = mglBufferMapStorageRemaining(map);
        NSUInteger availableBytes = buffer
            ? mglBufferMapVisibleBackingBytes(map, buffer.length)
            : 0u;
        BOOL needsIsolatedBinding =
            !buffer ||
            storageRemaining <= 0 ||
            bindOffset >= buffer.length ||
            availableBytes == 0 ||
            (requiredBytes > 0 && availableBytes < requiredBytes);

        MGLTessStageBufferBinding *binding = &bindings->slots[metalBindingIndex];
        binding->buffer = nil;
        binding->offset = 0u;
        binding->initialization_source = nil;
        binding->initialization_source_offset = 0u;
        binding->initialization_length = 0u;
        binding->valid = YES;
        if (!needsIsolatedBinding) {
            binding->buffer = buffer;
            binding->offset = bindOffset;
            /* The GL buffer's Metal backing is about to be staged in a
             * compute encoder: pin its snapshot-pool slot (P3). */
            mglNoteBufferEncoded(ptr);
            continue;
        }

        NSUInteger fallbackLength = MAX(requiredBytes, sizeof(uint32_t));
        MGLMetalBufferRef isolated = mglTessCreateBuffer(
            _device, fallbackLength, MTLResourceStorageModeShared);
        if (!isolated || !isolated.contents) {
            return false;
        }
        memset(isolated.contents, 0, fallbackLength);

        binding->buffer = isolated;
        binding->offset = 0u;
        if (buffer && availableBytes > 0) {
            binding->initialization_source = buffer;
            binding->initialization_source_offset = bindOffset;
            binding->initialization_length = MIN(availableBytes, fallbackLength);
        }

        BOOL writableResource =
            map->resource_type == _STORAGE_BUFFER_RES ||
            map->resource_type == _ATOMIC_COUNTER_RES;
        if (writableResource && buffer && availableBytes > 0 &&
            ![self recordStageBindingCopyBack:copyBacks
                                       atIndex:metalBindingIndex
                                     temporary:isolated
                                   destination:buffer
                             destinationBuffer:ptr
                            destinationOffset:bindOffset
                                        length:availableBytes]) {
            return false;
        }
    }

    Program *stageProgram = mglResolveProgramForStageFromState(ctx, stage);
    if (stageProgram &&
        stageProgram->modules[stage].needs_runtime_array_size_buffer) {
        bindings->size_buffer_index =
            mglRuntimeArraySizeBufferIndexForProgram(stageProgram, stage);
        uint32_t sizeConstants[kMGLMaxBufferSlots] = {0};
        for (GLuint i = 0; i < stageBufferMap.count; i++) {
            BufferMap *map = &stageBufferMap.buffers[i];
            if (!map->buf) continue;
            NSUInteger metalSlot = map->has_metal_binding
                ? (NSUInteger)map->metal_binding_index
                : (NSUInteger)map->buffer_base_index;
            if (metalSlot >= kMGLMaxBufferSlots ||
                metalSlot == bindings->size_buffer_index) {
                continue;
            }
            GLsizeiptr visibleSize = mglBufferMapVisibleSize(map);
            if (visibleSize > 0) {
                sizeConstants[metalSlot] = visibleSize > UINT32_MAX
                    ? UINT32_MAX : (uint32_t)visibleSize;
            }
        }
        bindings->size_buffer = mglTessCreateBufferWithBytes(
            _device, sizeConstants, sizeof(sizeConstants),
            MTLResourceStorageModeShared);
        if (!bindings->size_buffer) {
            return false;
        }
    }

    BOOL needsInitializationBlit = NO;
    for (NSUInteger i = 0; i < kMGLMaxBufferSlots; i++) {
        if (bindings->slots[i].initialization_length > 0) {
            needsInitializationBlit = YES;
            break;
        }
    }
    if (!needsInitializationBlit) {
        return true;
    }
    MGLRenderCppCommandBufferState commandState = {0};
    if (!mglRenderCommandBufferOwnerState(
            _renderPassManager.state->currentCommandBufferOwner,
            &commandState) ||
        commandState.status != MTLCommandBufferStatusNotEnqueued) {
        return false;
    }

    MGLRenderCppBufferCopyEntry copyEntries[kMGLMaxBufferSlots] = {0};
    uint32_t copyEntryCount = 0u;
    for (NSUInteger i = 0; i < kMGLMaxBufferSlots; i++) {
        MGLTessStageBufferBinding *binding = &bindings->slots[i];
        if (binding->initialization_length == 0) {
            continue;
        }
        copyEntries[copyEntryCount++] = (MGLRenderCppBufferCopyEntry){
            .source_buffer = (__bridge void *)binding->initialization_source,
            .source_offset = binding->initialization_source_offset,
            .destination_buffer = (__bridge void *)binding->buffer,
            .destination_offset = 0u,
            .length = binding->initialization_length,
        };
    }
    return mglTessEncodeBufferCopiesForOwner(
        _renderPassManager.state->currentCommandBufferOwner,
        copyEntries, copyEntryCount);
}

- (bool)bindPreparedTessStageBufferBindings:(const MGLTessStageBufferBindingList *)bindings
                           toComputeEncoder:(MGLMetalComputeCommandEncoderRef)computeCommandEncoder
                              executionPlan:(MGLRenderCppComputeExecutionPlan *)executionPlan
                               temporaries:(NSMutableArray *)temporaries
{
    MGL_ASSERT_GL_THREAD();
    (void)computeCommandEncoder;
    if (!bindings || !executionPlan) {
        return false;
    }
    for (NSUInteger i = 0; i < kMGLMaxBufferSlots; i++) {
        const MGLTessStageBufferBinding *binding = &bindings->slots[i];
        if (binding->valid) {
            if (!mglTessAppendComputeResourceOp(
                    executionPlan, temporaries, 0u, binding->buffer,
                    binding->offset, i)) {
                return false;
            }
        }
    }
    if (bindings->size_buffer) {
        if (!mglTessAppendComputeResourceOp(
                executionPlan, temporaries, 0u, bindings->size_buffer,
                0u, bindings->size_buffer_index)) {
            return false;
        }
    }
    return true;
}

/* Dispatch an AIR tessellation control shader (TCS) as a Metal compute kernel.
 * It writes tessellation factors to buffer(26) and per-patch output to buffer(27).
 * Indirect params (vertexCount, instanceCount) go in buffer(29).
 *
 * For shader_image_size tests the TCS kernel only needs storage images and
 * the tess-factor / indirect-param buffers — it has no vertex input.
 */
- (void)bindPointSizeParamsToComputeEncoder:(MGLMetalComputeCommandEncoderRef)computeEncoder
                                    program:(Program *)program
                                      stage:(int)stage
                              executionPlan:(MGLRenderCppComputeExecutionPlan *)executionPlan
                               temporaries:(NSMutableArray *)temporaries
{
    MGL_ASSERT_GL_THREAD();
    (void)computeEncoder;
    if (!executionPlan || !program ||
        stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return;
    }
    if (!program->uses_point_size_params) {
        return;
    }
    float pointSizeParams[2] = {
        ctx && MGL_STATE(ctx)->var.point_size > 0.0f ? MGL_STATE(ctx)->var.point_size : 1.0f,
        ctx && MGL_STATE(ctx)->caps.program_point_size ? 1.0f : 0.0f
    };
    (void)mglTessAppendComputeBytesOp(
        executionPlan, temporaries, pointSizeParams,
        sizeof(pointSizeParams), kMGLPointSizeParamBufferIndex);
}

- (MGLMetalBufferRef)newTCSStageInBufferForContext:(GLMContext)drawCtx
                                       program:(Program *)tcsProgram
                                         first:(GLint)first
                                         count:(GLsizei)count
                                     indexType:(GLenum)indexType
                                       indices:(const void *)indices
                                    baseVertex:(GLint)baseVertex
                                  baseInstance:(GLuint)baseInstance
                                 patchVertices:(GLuint)patchVertices
                                    patchCount:(GLuint)patchCount
                                     outStride:(NSUInteger *)outStride
{
    MGL_ASSERT_GL_THREAD();
    if (outStride) {
        *outStride = 0u;
    }
    if (!drawCtx || !tcsProgram || count <= 0) {
        return nil;
    }

    if (!tcsProgram->modules[_TESS_CONTROL_SHADER].metallib_bytes) {
        return nil;
    }

    MGLTCSStageInMember members[MAX_ATTRIBS];
    memset(members, 0, sizeof(members));
    NSUInteger tcsInStride = 0u;
    NSUInteger memberCount = 0u;
    /* AIR TCS shared record: position comes from attribute 0 and point size
     * and cull distances are initialized separately. */
    tcsInStride = MGL_AIR_PER_VERTEX_STRIDE;
    members[0].attribute = 0u;
    members[0].offset = 0u;
    members[0].size = 16u;
    members[0].componentBytes = 4u;
    members[0].components = 4u;
    members[0].baseType = MGLTCSStageInBaseFloat;
    memberCount = 1u;
    if (tcsInStride == 0u) {
        NSLog(@"MGL TESS WARNING: unable to compute TCS stage_in stride for program %u",
              (unsigned)tcsProgram->name);
        return nil;
    }

    NSUInteger tcsInVertices = (NSUInteger)patchCount * (NSUInteger)MAX(patchVertices, 1u);
    if (tcsInVertices < (NSUInteger)count) {
        tcsInVertices = (NSUInteger)count;
    }
    if (tcsInVertices == 0u || tcsInStride > NSUIntegerMax / tcsInVertices) {
        return nil;
    }

    VertexArray *vao = mglRendererGetValidatedVAO(drawCtx, "tcs.stage_in");
    if (!vao) {
        return nil;
    }

    const uint8_t *indexBytes = NULL;
    NSUInteger indexOffset = (NSUInteger)(uintptr_t)indices;
    NSUInteger indexStride = 0u;
    uint32_t restartIndex = 0u;
    bool primitiveRestart = false;
    if (indexType != 0u) {
        indexStride = mglGLIndexElementSize(indexType);
        if (indexStride == 0u || indexOffset > NSUIntegerMax - ((NSUInteger)count * indexStride)) {
            return nil;
        }
        Buffer *ebo = getElementBuffer(drawCtx);
        if (!ebo || ![self processBuffer:ebo]) {
            NSLog(@"MGL TESS WARNING: TCS indexed stage_in has no readable element buffer");
            return nil;
        }
        const uint8_t *eboBytes = mglRendererReadableBufferBytes(ebo);
        NSUInteger bytesNeeded = (NSUInteger)count * indexStride;
        if (!eboBytes || indexOffset > (NSUInteger)ebo->size || ((NSUInteger)ebo->size - indexOffset) < bytesNeeded) {
            NSLog(@"MGL TESS WARNING: TCS indexed stage_in element range OOB offset=%lu needed=%lu size=%lld",
                  (unsigned long)indexOffset,
                  (unsigned long)bytesNeeded,
                  (long long)ebo->size);
            return nil;
        }
        indexBytes = eboBytes + indexOffset;
        primitiveRestart = mglPrimitiveRestartIndexForType(drawCtx, indexType, &restartIndex);
    }

    NSUInteger tcsInSize = tcsInStride * tcsInVertices;
    MGLMetalBufferRef stageInBuffer = mglTessCreateBuffer(
        _device, tcsInSize, MTLResourceStorageModeShared);
    if (!stageInBuffer) {
        return nil;
    }
    memset(stageInBuffer.contents, 0, tcsInSize);
    for (NSUInteger v = 0; v < tcsInVertices; v++) {
        float one = 1.0f;
        uint8_t *record = (uint8_t *)stageInBuffer.contents + v * tcsInStride;
        memcpy(record + MGL_AIR_PER_VERTEX_POINT_SIZE_OFFSET,
               &one, sizeof(one));
        for (NSUInteger distance = 0;
             distance < MGL_AIR_PER_VERTEX_CULL_DISTANCE_COUNT; distance++) {
            memcpy(record + MGL_AIR_PER_VERTEX_CULL_DISTANCE_OFFSET +
                       distance * sizeof(float),
                   &one, sizeof(one));
        }
    }

    if (memberCount == 0u) {
        if (outStride) {
            *outStride = tcsInStride;
        }
        return stageInBuffer;
    }

    for (NSUInteger v = 0; v < tcsInVertices; v++) {
        if (v >= (NSUInteger)count) {
            continue;
        }

        int64_t vertexIndex64 = (int64_t)first + (int64_t)v;
        if (indexBytes) {
            uint32_t rawIndex = mglReadGLIndexValue(indexBytes, indexType, v);
            if (primitiveRestart && rawIndex == restartIndex) {
                continue;
            }
            vertexIndex64 = (int64_t)rawIndex + (int64_t)baseVertex;
        }
        if (vertexIndex64 < 0) {
            continue;
        }

        uint8_t *dstVertex = (uint8_t *)stageInBuffer.contents + (v * tcsInStride);
        for (NSUInteger m = 0; m < memberCount; m++) {
            const MGLTCSStageInMember *member = &members[m];
            if (member->attribute >= MAX_ATTRIBS ||
                member->offset >= tcsInStride ||
                member->size > tcsInStride - member->offset) {
                continue;
            }

            double values[4] = {0.0, 0.0, 0.0, 1.0};
            const VertexAttrib *attrib = &vao->attrib[member->attribute];
            MGLResolvedVertexAttribBinding resolved = {0};
            bool hasBinding = mglRendererResolveVertexAttribBinding(drawCtx,
                                                                    vao,
                                                                    member->attribute,
                                                                    "tcs.stage_in",
                                                                    &resolved);
            bool useCurrentValue =
                ((vao->enabled_attribs & (0x1u << member->attribute)) == 0u) &&
                !(vao->enabled_attribs == 0u && hasBinding);

            if (useCurrentValue) {
                uint8_t currentBytes[16];
                if (mglRendererBuildCurrentVertexAttribBytes(drawCtx,
                                                             member->attribute,
                                                             attrib,
                                                             currentBytes) > 0u) {
                    for (GLuint c = 0; c < MIN(attrib->size, 4u); c++) {
                        values[c] = mglDecodeVertexAttribComponent(currentBytes,
                                                                   attrib->type,
                                                                   attrib->normalized,
                                                                   c);
                    }
                }
            } else if (hasBinding) {
                Buffer *vbo = resolved.buffer;
                if (vbo && [self processBuffer:vbo]) {
                    const uint8_t *vboBytes = mglRendererReadableBufferBytes(vbo);
                    NSUInteger elementBytes = mglVertexAttribElementBytes(attrib->type, attrib->size);
                    NSUInteger stride = resolved.stride > 0u ? (NSUInteger)resolved.stride : elementBytes;
                    NSUInteger attribIndex = (NSUInteger)vertexIndex64;
                    if (resolved.divisor > 0u) {
                        attribIndex = (NSUInteger)(baseInstance / resolved.divisor);
                    }
                    if (vboBytes && elementBytes > 0u && stride > 0u &&
                        resolved.binding_offset >= 0 && resolved.relativeoffset >= 0) {
                        NSUInteger baseOffset = (NSUInteger)resolved.binding_offset + (NSUInteger)resolved.relativeoffset;
                        if (attribIndex <= (NSUIntegerMax - baseOffset) / stride) {
                            NSUInteger vertexOffset = baseOffset + attribIndex * stride;
                            if (vertexOffset <= (NSUInteger)vbo->size &&
                                ((NSUInteger)vbo->size - vertexOffset) >= elementBytes) {
                                GLboolean effectiveNormalized = attrib->normalized;
                                const uint8_t *src = vboBytes + vertexOffset;
                                for (GLuint c = 0; c < MIN(attrib->size, 4u); c++) {
                                    values[c] = mglDecodeVertexAttribComponent(src,
                                                                               attrib->type,
                                                                               effectiveNormalized,
                                                                               c);
                                }
                            }
                        }
                    }
                }
            }

            for (GLuint c = 0; c < member->components && c < 4u; c++) {
                mglWriteTCSStageInComponent(dstVertex, member, c, values[c]);
            }
        }
    }

    if (outStride) {
        *outStride = tcsInStride;
    }
    return stageInBuffer;
}

-(bool) dispatchTessControlShader:(GLMContext) glm_ctx
                          program:(Program *) tcsProgram
                         contract:(const MGLAIRTessDrawContract *) contract
{
    if (!tcsProgram || !glm_ctx || !contract) {
        return false;
    }

    Shader *tcsShader = tcsProgram->shader_slots[_TESS_CONTROL_SHADER];
    if (!tcsShader || !tcsProgram->modules[_TESS_CONTROL_SHADER].mtl_function) {
        NSLog(@"MGL TESS WARNING: TCS program %u has no compiled function", tcsProgram->name);
        return false;
    }

    /* Create compute pipeline state for TCS kernel. */
    void *tcsPipelineHandle = NULL;
    char tcsPipelineError[512] = {0};
    int tcsPipelineResult = mglGetOrCreateProgramComputePipeline(
        tcsProgram, _TESS_CONTROL_SHADER, &tcsPipelineHandle,
        tcsPipelineError, sizeof(tcsPipelineError));
    MGLMetalComputePipelineStateRef tcsPipeline =
        tcsPipelineResult == 0 && tcsPipelineHandle
            ? (__bridge_transfer MGLMetalComputePipelineStateRef)tcsPipelineHandle
            : nil;
    if (!tcsPipeline) {
        NSLog(@"MGL TESS ERROR: failed to create TCS compute pipeline for program %u: %s",
              tcsProgram->name,
              tcsPipelineError[0] ? tcsPipelineError : "unknown error");
        return false;
    }

    /* PASS 1: Pre-resolve all Metal textures that the TCS kernel needs.
     * This must happen BEFORE we open a compute encoder, because lazy
     * Metal texture creation (bindMTLTexture:) may open its own blit
     * encoder on the command buffer, and Metal forbids two encoders
     * on the same command buffer simultaneously.  End any active render
     * encoder first for the same reason. */
    if (mglRenderCppRenderEncoderOwnerHasCurrent(
            _renderPassManager.state->currentRenderEncoderOwner) == 1) {
        [self endRenderEncoding];
    }

    /* Ensure a writable command buffer exists.  The GL_PATCHES path returns
     * before processGLState() (which normally creates the command buffer),
     * and prior operations (glBufferData, glEndQuery, etc.) may have
     * committed the previous command buffer. */
    MGLRenderCppCommandBufferState commandState = {0};
    if (!mglRenderCommandBufferOwnerState(
            _renderPassManager.state->currentCommandBufferOwner,
            &commandState) ||
        commandState.status >= MTLCommandBufferStatusCommitted) {
        if (![self newCommandBuffer]) {
            NSLog(@"MGL TESS ERROR: failed to create command buffer for TCS dispatch");
            return false;
        }
    }

    GLuint tcsImgCount = mglRendererGetProgramBindingCount(ctx, _TESS_CONTROL_SHADER, _STORAGE_IMAGE_RES);
    for (GLuint i = 0; i < tcsImgCount; i++) {
        MGLShaderResource *resource = NULL;
        if (tcsProgram &&
            i < tcsProgram->shader_resources_list[_TESS_CONTROL_SHADER][_STORAGE_IMAGE_RES].count) {
            resource = &tcsProgram->shader_resources_list[_TESS_CONTROL_SHADER][_STORAGE_IMAGE_RES].list[i];
        }
        if (mglShouldSkipStageTextureResource(tcsProgram,
                                              _TESS_CONTROL_SHADER,
                                              _STORAGE_IMAGE_RES,
                                              resource)) {
            continue;
        }
        GLuint glUnit = resource ? resource->gl_binding
                                 : mglRendererGetProgramGLBinding(ctx, _TESS_CONTROL_SHADER, _STORAGE_IMAGE_RES, (int)i);
        if (glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = MGL_STATE(ctx)->image_units[glUnit].tex;
        if (ptr && !ptr->mtl_data) {
            [self bindMTLTexture:ptr];
        }
    }

    MGLStageBindingCopyBackList stageCopyBacks = {0};
    MGLTessStageBufferBindingList stageBufferBindings = {0};
    if (![self prepareTessStageBufferBindings:&stageBufferBindings
                                         stage:_TESS_CONTROL_SHADER
                                     copyBacks:&stageCopyBacks]) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    MGLRenderCppComputeExecutionPlan executionPlan = {0};
    NSMutableArray *executionTemporaries = [NSMutableArray array];
    MGLMetalComputeCommandEncoderRef computeEncoder = nil;
    executionPlan.pipeline = (__bridge void *)tcsPipeline;

    /* PASS 2: Bind storage images for TCS stage. */
    for (GLuint i = 0; i < tcsImgCount; i++) {
        MGLShaderResource *resource = NULL;
        if (tcsProgram &&
            i < tcsProgram->shader_resources_list[_TESS_CONTROL_SHADER][_STORAGE_IMAGE_RES].count) {
            resource = &tcsProgram->shader_resources_list[_TESS_CONTROL_SHADER][_STORAGE_IMAGE_RES].list[i];
        }
        if (mglShouldSkipStageTextureResource(tcsProgram,
                                              _TESS_CONTROL_SHADER,
                                              _STORAGE_IMAGE_RES,
                                              resource)) {
            continue;
        }
        GLuint metalSlot = resource ? mglMetalResourceSlot(resource)
                                    : mglRendererGetProgramBinding(ctx, _TESS_CONTROL_SHADER, _STORAGE_IMAGE_RES, (int)i);
        GLuint glUnit = resource ? resource->gl_binding
                                 : mglRendererGetProgramGLBinding(ctx, _TESS_CONTROL_SHADER, _STORAGE_IMAGE_RES, (int)i);
        if (metalSlot >= TEXTURE_UNITS || glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = MGL_STATE(ctx)->image_units[glUnit].tex;
        MGLMetalTextureRef texture = nil;
        if (ptr) {
            texture = (__bridge MGLMetalTextureRef)(ptr->mtl_data);
            GLuint imgLevel = MGL_STATE(ctx)->image_units[glUnit].level;
            if (imgLevel > 0u && texture) {
                NSUInteger sliceCount = texture.arrayLength;
                if (texture.textureType == MTLTextureTypeCube ||
                    texture.textureType == MTLTextureTypeCubeArray) {
                    sliceCount = texture.arrayLength * 6u;
                }
                MGLMetalTextureRef levelView =
                    mglTessCreateTextureLevelView(
                        texture, imgLevel, sliceCount);
                if (levelView) {
                    texture = levelView;
                }
            }
        }
        if (!mglTessPlanTextureOrBind(
                &executionPlan,
                executionTemporaries, computeEncoder, texture, metalSlot)) {
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }
    }

    /* Also bind sampled (read-only) images for TCS stage. */
    GLuint tcsSampledCount = mglRendererGetProgramBindingCount(ctx, _TESS_CONTROL_SHADER, _SAMPLED_IMAGE_RES);
    for (GLuint i = 0; i < tcsSampledCount; i++) {
        MGLShaderResource *resource = NULL;
        if (tcsProgram &&
            i < tcsProgram->shader_resources_list[_TESS_CONTROL_SHADER][_SAMPLED_IMAGE_RES].count) {
            resource = &tcsProgram->shader_resources_list[_TESS_CONTROL_SHADER][_SAMPLED_IMAGE_RES].list[i];
        }
        if (mglShouldSkipStageTextureResource(tcsProgram,
                                              _TESS_CONTROL_SHADER,
                                              _SAMPLED_IMAGE_RES,
                                              resource)) {
            continue;
        }
        GLuint metalSlot = resource ? mglMetalResourceSlot(resource)
                                    : mglRendererGetProgramBinding(ctx, _TESS_CONTROL_SHADER, _SAMPLED_IMAGE_RES, (int)i);
        GLuint glUnit = resource ? resource->gl_binding
                                 : mglRendererGetProgramGLBinding(ctx, _TESS_CONTROL_SHADER, _SAMPLED_IMAGE_RES, (int)i);
        if (metalSlot >= TEXTURE_UNITS || glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = MGL_STATE(ctx)->active_textures[glUnit];
        if (ptr && !ptr->mtl_data) {
            [self bindMTLTexture:ptr];
        }
        MGLMetalTextureRef texture = ptr ? (__bridge MGLMetalTextureRef)(ptr->mtl_data) : nil;
        if (!mglTessPlanTextureOrBind(
                &executionPlan,
                executionTemporaries, computeEncoder, texture, metalSlot)) {
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }
        if (resource && resource->has_combined_sampler) {
            MGLMetalSamplerStateRef sampler = nil;
            if (MGL_STATE(ctx)->texture_samplers[glUnit]) {
                Sampler *glSampler = MGL_STATE(ctx)->texture_samplers[glUnit];
                if (glSampler->dirty_bits && glSampler->mtl_data) {
                    mglSafeReleaseMetalObj((void **)&glSampler->mtl_data);
                }
                if (!glSampler->mtl_data && ptr) {
                    glSampler->mtl_data = (void *)CFBridgingRetain(
                        [self createMTLSamplerForTexParam:&glSampler->params
                                                  target:ptr->target]);
                    glSampler->dirty_bits = 0;
                }
                sampler = (__bridge MGLMetalSamplerStateRef)(glSampler->mtl_data);
            } else if (ptr && ptr->params.mtl_data) {
                sampler = (__bridge MGLMetalSamplerStateRef)(ptr->params.mtl_data);
            }
            if (!sampler) {
                sampler = mglTessCreateSampler(
                    _device, [MTLSamplerDescriptor new]);
            }
            if (sampler) {
                if (!mglTessPlanSamplerOrBind(
                        &executionPlan,
                        executionTemporaries, computeEncoder, sampler,
                        mglMetalCombinedSamplerSlot(resource))) {
                    [self clearStageBindingCopyBacks:&stageCopyBacks];
                    return false;
                }
            }
        }
    }

    /* Bind stage buffers (UBO, SSBO, atomic counters) for TCS. */
    if (![self bindPreparedTessStageBufferBindings:&stageBufferBindings
                                  toComputeEncoder:computeEncoder
                                     executionPlan:&executionPlan
                                      temporaries:executionTemporaries]) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    [self bindPointSizeParamsToComputeEncoder:computeEncoder
                                      program:tcsProgram
                                        stage:_TESS_CONTROL_SHADER
                                executionPlan:&executionPlan
                                 temporaries:executionTemporaries];

    /* Create indirect params buffer (buffer 29).
     * spvIndirectParams[0] = vertexCount, [1] = instanceCount. */
    const GLuint patchVertices = MAX(1u, contract->patch_vertices);
    const GLuint vertexCount = contract->vertex_count;
    const GLuint instanceCount = MAX(1u, contract->instance_count);

    /* Create TCS per-vertex output buffer (buffer 28 = spvOut).
     * TCS writes: spvOut[gl_PrimitiveID * outputVertices + invocationID]
     * where outputVertices = tess_control_output_vertices (layout(vertices=N) out).
     * Compute the per-vertex stride from the TCS stage output resources. */
    GLuint tcsOutVertices = tcsProgram->tess_control_output_vertices;
    if (tcsOutVertices == 0) tcsOutVertices = patchVertices;

    NSUInteger tcsOutStride = mglAIRPerVertexStrideForResources(
        &tcsProgram->shader_resources_list[_TESS_CONTROL_SHADER]
                                                 [_STAGE_OUTPUT_RES]);
    _tessellation.tcsOutputStride = tcsOutStride;
    _tessellation.tcsOutVertices = tcsOutVertices;

    GLuint patchCountTC = vertexCount / patchVertices;
    if (patchCountTC == 0u) patchCountTC = 1u;
    NSUInteger tcsOutSize = (NSUInteger)patchCountTC * tcsOutVertices * tcsOutStride;
    _tessellation.tcsOutputBuffer = mglTessCreateBuffer(
        _device, tcsOutSize, MTLResourceStorageModeShared);
    if (!_tessellation.tcsOutputBuffer || !_tessellation.tcsOutputBuffer.contents) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    memset(_tessellation.tcsOutputBuffer.contents, 0, tcsOutSize);
    _tessellation.tcsOutputOffset = 0u;
    /* TCS stage output (spvOut) binds at slot 28 — the same numeric slot as
     * the TES patch-info constant, reused across disjoint encoders.  The
     * TCS kernel's fixed ABI is stage_in(24) factors(26) patchOut(27)
     * stageOut(28) indirect(29); see mgl_air_tess_abi.h. */
    if (!mglTessPlanBufferOrBind(
            &executionPlan,
            executionTemporaries, computeEncoder,
            _tessellation.tcsOutputBuffer, 0,
            MGL_AIR_TESS_SLOT_TCS_OUTPUT)) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    /* Create TCS per-patch output buffer (buffer 27 = spvPatchOut).
     * Patch varyings use the same stable location ABI as other stage
     * interfaces: one 16-byte slot per location. */
    NSUInteger tcsPatchStride = 16u;
    if (tcsProgram) {
        /* Per-patch outputs share _STAGE_OUTPUT_RES with
         * per-vertex outputs; SpvDecorationPatch is reflected as is_per_patch. */
        MGLShaderResourceList *outs =
            &tcsProgram->shader_resources_list[_TESS_CONTROL_SHADER][_STAGE_OUTPUT_RES];
        tcsPatchStride = mglAIRPatchVaryingStride(outs);
    }
    NSUInteger tcsPatchSize = (NSUInteger)patchCountTC * tcsPatchStride;
    MGLMetalBufferRef tcsPatchOutBuffer = mglTessCreateBuffer(
        _device, tcsPatchSize, MTLResourceStorageModeShared);
    (void)mglRendererBackendSetTcsPatchOutBuffer(
        _backend, (__bridge void *)tcsPatchOutBuffer);
    if (!tcsPatchOutBuffer || !tcsPatchOutBuffer.contents) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    memset(tcsPatchOutBuffer.contents, 0, tcsPatchSize);
    if (!mglTessPlanBufferOrBind(
            &executionPlan,
            executionTemporaries, computeEncoder,
            tcsPatchOutBuffer, 0,
            MGL_AIR_TESS_SLOT_PATCH_OUT)) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    GLuint indirectParams[2] = { patchVertices, instanceCount };
    MGLMetalBufferRef indirectBuf = mglTessCreateBufferWithBytes(
        _device, indirectParams, sizeof(indirectParams),
        MTLResourceStorageModeShared);
    if (!indirectBuf) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    if (!mglTessPlanBufferOrBind(
            &executionPlan,
            executionTemporaries, computeEncoder, indirectBuf, 0,
            MGL_AIR_TESS_SLOT_INDIRECT)) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    /* Create tessellation factor buffer (buffer 26).
     * The compute path always uses the quad-half layout: 4 edge + 2 inner
     * half-floats = 12 bytes/patch (MGL_AIR_TESS_FACTOR_QUAD_HALF_BYTES). */
    const GLuint patchCount = MAX(1u, contract->patch_count);
    NSUInteger tessFactorSize =
        (NSUInteger)patchCount * MGL_AIR_TESS_FACTOR_QUAD_HALF_BYTES;
    MGLMetalBufferRef tessFactorBuf = mglTessCreateBuffer(
        _device, tessFactorSize, MTLResourceStorageModeShared);
    if (!tessFactorBuf || !tessFactorBuf.contents) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    memset(tessFactorBuf.contents, 0, tessFactorSize);
    if (!mglTessPlanBufferOrBind(
            &executionPlan,
            executionTemporaries, computeEncoder, tessFactorBuf, 0,
            MGL_AIR_TESS_SLOT_TESS_FACTOR)) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    NSUInteger tcsInStride = 0u;
    MGLMetalBufferRef tcsStageInBuffer =
        (__bridge MGLMetalBufferRef)
            mglRendererBackendGetTessVertexCaptureBuffer(_backend);
    NSUInteger tcsStageInOffset = _tessellation.tessVertexCaptureOffset;
    if (tcsStageInBuffer) {
        tcsInStride = mglAIRPerVertexStrideForResources(
            &tcsProgram->shader_resources_list[_TESS_CONTROL_SHADER]
                                                     [_STAGE_INPUT_RES]);
    } else {
        tcsStageInBuffer =
            [self newTCSStageInBufferForContext:glm_ctx
                                        program:tcsProgram
                                          first:contract->first
                                          count:(GLsizei)contract->vertex_count
                                      indexType:contract->index_type
                                        indices:(const void *)(uintptr_t)contract->index_source
                                     baseVertex:contract->base_vertex
                                   baseInstance:contract->base_instance
                                  patchVertices:patchVertices
                                     patchCount:patchCount
                                      outStride:&tcsInStride];
        tcsStageInOffset = 0u;
    }
    if (!tcsStageInBuffer) {
        NSLog(@"MGL TESS WARNING: failed to pack TCS stage_in buffer for program %u",
              tcsProgram ? (unsigned)tcsProgram->name : 0u);
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    if (!mglTessPlanBufferOrBind(
            &executionPlan,
            executionTemporaries, computeEncoder, tcsStageInBuffer,
            tcsStageInOffset, MGL_AIR_TESS_SLOT_TCS_STAGE_IN)) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    /* Dispatch: one threadgroup per patch, tcsOutVertices threads per threadgroup (one thread per TCS output vertex = gl_InvocationID). */
    MTLSize threadgroups = MTLSizeMake(patchCount, 1, 1);
    MTLSize threadsPerTG = MTLSizeMake(tcsOutVertices, 1, 1);
    if (!mglTessPlanDispatchOrBind(
            &executionPlan, computeEncoder,
            threadgroups, threadsPerTG)) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    {
        MGLRenderCppCopyBackEntry copyBackEntries[kMGLMaxBufferSlots] = {0};
        uint32_t copyBackEntryCount = 0u;
        for (NSUInteger slot = 0; slot < kMGLMaxBufferSlots; slot++) {
            MGLStageBindingCopyBack *entry = &stageCopyBacks.slots[slot];
            if (entry->length == 0u) continue;
            copyBackEntries[copyBackEntryCount++] =
                (MGLRenderCppCopyBackEntry){
                    .temporary = (__bridge void *)entry->temporary,
                    .destination = (__bridge void *)entry->destination,
                    .destination_buffer = entry->destination_buffer,
                    .destination_offset = entry->destination_offset,
                    .length = entry->length,
                };
        }
        executionPlan.barrier_scope = MGL_RENDER_CPP_COMPUTE_BARRIER_BUFFERS;
        MGLRenderCppComputeExecutionResult executionResult = {0};
        char executionError[256] = {0};
        if (mglRenderCppExecuteComputeExecutionPlan(
                _renderPassManager.state->currentCommandBufferOwner,
                _gpuRecovery.commandRecoveryOwner,
                &executionPlan, copyBackEntries, copyBackEntryCount, 1u,
                &executionResult, executionError,
                sizeof(executionError)) != 0) {
            if (executionResult.transaction.device_reset_requested) {
                atomic_store_explicit(&_deviceResetRequested, true,
                                      memory_order_release);
            }
            NSLog(@"MGL TESS ERROR: C++ TCS execution failed: %s",
                  executionError[0] ? executionError : "unknown error");
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }
        [self clearStageBindingCopyBacks:&stageCopyBacks];
    }

    /* Save tess factor buffer for TES patch-draw path. */
    _tessellation.tessFactorBuffer = tessFactorBuf;

    return true;
}

/* P4.5 (item 1141/887): TES XFB field byte size for a GL type (FLOAT/INT/
 * UINT + vec2/3/4; 0 for unsupported).  Single source of truth is
 * mglRenderCppTESXFBFieldByteSize (mgl_render_cpp.cpp) — kept in lockstep
 * with the packed writes injected by mglFixMSLTesAsComputeKernel; a zero
 * result means the renderer cannot prove the write stride. */
static NSUInteger mglTESXFBFieldByteSize(GLenum glType)
{
    return (NSUInteger)mglRenderCppTESXFBFieldByteSize((uint64_t)glType);
}

/* Keep this layout calculation in lockstep with the packed writes injected by
 * mglFixMSLTesAsComputeKernel.  A zero result means the renderer cannot prove
 * the write stride and must not copy temporary capture data into the GL store.
 * P4.5 (item 1141/887): single source of truth is
 * mglRenderCppTESXFBVertexStride (mgl_render_cpp.cpp) — this shell keeps
 * the two XFB stride call sites unchanged. */
static NSUInteger mglTESXFBVertexStride(const Program *program)
{
    return (NSUInteger)mglRenderCppTESXFBVertexStride((const void *)program);
}

/* P4.5 (item 1141/887): overflow-checked product for tessellation size
 * math.  Single source of truth is mglRenderCppCheckedProduct
 * (mgl_render_cpp.cpp) — this shell keeps the eight call sites unchanged. */
static bool mglCheckedNSUIntegerProduct(NSUInteger a,
                                        NSUInteger b,
                                        NSUInteger *result)
{
    uint64_t out = 0u;
    if (mglRenderCppCheckedProduct((uint64_t)a, (uint64_t)b, &out) != 0) {
        return false;
    }
    *result = (NSUInteger)out;
    return true;
}

/* GL 4.6 §11.2.2.2 subdivision count rounding: fractional_even rounds up
 * to the next even (min 2), fractional_odd to the next odd; integer (and
 * the default) keep ceil(level).  Must match the AIR TES compute generator
 * and the native-tess query accounting. */
/* Per-patch expanded item count for the isolines/point-mode TES kernel.
 * Must stay in lockstep with the u/v decomposition injected by
 * mgl_air_backend.cpp (isTESCompute pre-main block).  Returns 0 when the
 * factor record is missing (caller falls back to 1). */

/* P4.5 (item 1141/887): GL 4.6 §11.2.2.2 subdivision-count rounding.
 * Single source of truth is mglRenderCppTessRoundLevelForSpacing
 * (mgl_render_cpp.cpp) — this shell keeps the six native per-patch counting
 * call sites unchanged. */
static GLuint mglTessRoundLevelForSpacing(GLenum spacing, GLuint ceilLevel)
{
    return (GLuint)mglRenderCppTessRoundLevelForSpacing(
        (uint32_t)spacing, (uint32_t)ceilLevel);
}

static GLuint mglAIRTessEvalItemsPerPatch(const Program *tesProgram,
                                          const void *factorRecord)
{
    /* P4.5 (item 1141/887): 逐 patch 展开 item 计数（isolines/point-mode
     * TES 核的 u/v 分解 lockstep + discard 判定 + spacing 取整）在 C++
     * （mglRenderCppTessEvalItemsPerPatch，纯数据变换，两门共用）。 */
    return (GLuint)mglRenderCppTessEvalItemsPerPatch(
        factorRecord,
        (uint32_t)(tesProgram ? tesProgram->tess_gen_mode : GL_TRIANGLES),
        (uint32_t)(tesProgram ? tesProgram->tess_gen_spacing : 0),
        (uint32_t)(tesProgram ? tesProgram->tess_gen_point_mode : 0));
}

/* Isolines / point-mode TES: expand one vertex record per work item with
 * the AIR TES compute kernel (backend ABI: stage_in(24) factors(26)
 * patchInputs(27) stageOut(28) indirect(29), one dispatch per patch), then
 * rasterize through the passthrough vertex stage as lines / points.
 * Each patch owns a contiguous item span; per-patch item counts differ,
 * so the runtime dispatches per patch with the patch id and output base in
 * the contract buffer (slot 29: {patch_id, gl_in_vertices, items,
 * output_item_base}). */
- (BOOL)dispatchAIRTessEvalCompute:(GLMContext)glm_ctx
                          program:(Program *)tesProgram
                         contract:(const MGLAIRTessDrawContract *)contract
                       patchCount:(GLuint)patchCount
                    instanceCount:(GLsizei)instanceCount
                     baseInstance:(GLuint)baseInstance
{
    if (!tesProgram || !glm_ctx || !contract || patchCount == 0u ||
        instanceCount <= 0) {
        return false;
    }

    Shader *tesShader = tesProgram->shader_slots[_TESS_EVALUATION_SHADER];
    if (!tesShader || !tesProgram->modules[_TESS_EVALUATION_SHADER].mtl_function) {
        NSLog(@"MGL TESS WARNING: TES program %u has no compiled function",
              tesProgram->name);
        return false;
    }

    void *tesPipelineHandle = NULL;
    char tesPipelineError[512] = {0};
    int tesPipelineResult = mglGetOrCreateProgramComputePipeline(
        tesProgram, _TESS_EVALUATION_SHADER, &tesPipelineHandle,
        tesPipelineError, sizeof(tesPipelineError));
    MGLMetalComputePipelineStateRef tesPipeline =
        tesPipelineResult == 0 && tesPipelineHandle
            ? (__bridge_transfer MGLMetalComputePipelineStateRef)tesPipelineHandle
            : nil;
    if (!tesPipeline) {
        NSLog(@"MGL TESS ERROR: failed to create TES compute pipeline for program %u: %s",
              tesProgram->name,
              tesPipelineError[0] ? tesPipelineError : "unknown error");
        return false;
    }

    /* Inputs: gl_in is the post-TCS control point stream (or the VS capture
     * when there is no TCS, which the draw path already aliased into
     * tcsOutputBuffer).  Factors and per-patch inputs come from the TCS
     * dispatch (or defaults). */
    MGLMetalBufferRef glInBuffer = _tessellation.tcsOutputBuffer;
    NSUInteger glInOffset = _tessellation.tcsOutputOffset;
    NSUInteger glInStride = _tessellation.tcsOutputStride;
    GLuint glInVertices = _tessellation.tcsOutVertices;
    if (!glInBuffer) {
        glInBuffer = (__bridge MGLMetalBufferRef)
            mglRendererBackendGetTessVertexCaptureBuffer(_backend);
        glInOffset = _tessellation.tessVertexCaptureOffset;
        glInStride = contract->per_vertex_out_stride;
        glInVertices = MAX(1u, contract->patch_vertices);
    }
    if (!glInBuffer || !_tessellation.tessFactorBuffer) {
        NSLog(@"MGL TESS ERROR: missing TES compute inputs program=%u",
              (unsigned)tesProgram->name);
        return false;
    }
    MGLMetalBufferRef controlPointIndexBuffer =
        (__bridge MGLMetalBufferRef)
            mglRendererBackendGetTessControlPointIndexBuffer(_backend);
    if (_tessellation.tessIndexedDraw) {
        /* Indexed draws: the capture is a sparse [instance][vertex_id]
         * stream read through the gather stream in the kernel (slot 30 +
         * params slot 25); the instance offset math below does not apply. */
        if (!controlPointIndexBuffer ||
            _tessellation.tessInstanceRecords == 0u) {
            NSLog(@"MGL TESS ERROR: indexed TES compute missing gather "
                  "program=%u", (unsigned)tesProgram->name);
            return false;
        }
    }
    if (glInStride < MGL_AIR_PER_VERTEX_STRIDE) {
        glInStride = MGL_AIR_PER_VERTEX_STRIDE;
    }
    if (glInVertices == 0u) glInVertices = MAX(1u, contract->patch_vertices);
    const BOOL glInFromTCS = (glInBuffer == _tessellation.tcsOutputBuffer);
    const NSUInteger glInInstanceStride =
        (glInFromTCS || _tessellation.tessIndexedDraw)
            ? 0u
            : _tessellation.tessInstanceRecords * glInStride;
    if (glInFromTCS && instanceCount > 1) {
        /* The TCS kernel has no instance dimension (its dispatch is
         * patchCount threads over a single spvOut span), so a second
         * instance would read the wrong control points.  Reject like the
         * native path instead of rendering garbage. */
        NSLog(@"MGL TESS ERROR: TCS + instanced TES compute unsupported "
              "program=%u instances=%d",
              tesProgram->name, instanceCount);
        return false;
    }

    /* Compute per-patch item counts and the per-instance total. */
    const uint16_t *factorBytes =
        (const uint16_t *)_tessellation.tessFactorBuffer.contents;
    NSUInteger factorByteCount =
        (NSUInteger)patchCount * MGL_AIR_TESS_FACTOR_QUAD_HALF_BYTES;
    if ((NSUInteger)_tessellation.tessFactorBuffer.length < factorByteCount) {
        NSLog(@"MGL TESS ERROR: factor buffer too small for %u patches",
              (unsigned)patchCount);
        return false;
    }
    const GLuint instanceCountU = (GLuint)instanceCount;
    uint64_t itemsPerInstance = 0u;
    for (GLuint p = 0u; p < patchCount; p++) {
        const void *record =
            (const void *)((const uint8_t *)factorBytes +
                           (NSUInteger)p *
                               MGL_AIR_TESS_FACTOR_QUAD_HALF_BYTES);
        GLuint items = mglAIRTessEvalItemsPerPatch(tesProgram, record);
        if (items == 0u) items = 1u;
        itemsPerInstance += (uint64_t)items;
    }
    if (itemsPerInstance == 0u ||
        itemsPerInstance > 0xffffffffull) {
        NSLog(@"MGL TESS ERROR: TES compute item count overflow program=%u",
              (unsigned)tesProgram->name);
        return false;
    }
    const GLuint itemsPerInstanceU = (GLuint)itemsPerInstance;

    NSUInteger outStride = mglAIRPerVertexStrideForResources(
        &tesProgram->shader_resources_list[_TESS_EVALUATION_SHADER]
                                             [_STAGE_OUTPUT_RES]);
    if (outStride < MGL_AIR_PER_VERTEX_STRIDE) {
        outStride = MGL_AIR_PER_VERTEX_STRIDE;
    }
    NSUInteger instanceBytes = 0u;
    NSUInteger outSize = 0u;
    if (!mglCheckedNSUIntegerProduct(itemsPerInstanceU, outStride,
                                     &instanceBytes) ||
        !mglCheckedNSUIntegerProduct(instanceBytes, instanceCountU, &outSize)) {
        NSLog(@"MGL TESS ERROR: TES compute output size overflow program=%u",
              (unsigned)tesProgram->name);
        return false;
    }
    MGLMetalBufferRef outBuffer = mglTessCreateBuffer(
        _device, outSize, MTLResourceStorageModeShared);
    if (!outBuffer || !outBuffer.contents) {
        NSLog(@"MGL TESS ERROR: failed to allocate TES compute output "
              "(%lu bytes) program=%u",
              (unsigned long)outSize, (unsigned)tesProgram->name);
        return false;
    }
    memset(outBuffer.contents, 0, outSize);

    /* PASS 1: pre-resolve textures before opening the compute encoder. */
    if (mglRenderCppRenderEncoderOwnerHasCurrent(
            _renderPassManager.state->currentRenderEncoderOwner) == 1) {
        [self endRenderEncoding];
    }
    MGLRenderCppCommandBufferState commandState = {0};
    if (!mglRenderCommandBufferOwnerState(
            _renderPassManager.state->currentCommandBufferOwner,
            &commandState) ||
        commandState.status >= MTLCommandBufferStatusCommitted) {
        if (![self newCommandBuffer]) {
            NSLog(@"MGL TESS ERROR: failed to create command buffer for TES compute");
            return false;
        }
    }

    GLuint tesImgCount = mglRendererGetProgramBindingCount(ctx, _TESS_EVALUATION_SHADER, _STORAGE_IMAGE_RES);
    for (GLuint i = 0; i < tesImgCount; i++) {
        MGLShaderResource *resource = NULL;
        if (i <
            tesProgram->shader_resources_list[_TESS_EVALUATION_SHADER][_STORAGE_IMAGE_RES].count) {
            resource =
                &tesProgram->shader_resources_list[_TESS_EVALUATION_SHADER][_STORAGE_IMAGE_RES].list[i];
        }
        if (mglShouldSkipStageTextureResource(tesProgram,
                                              _TESS_EVALUATION_SHADER,
                                              _STORAGE_IMAGE_RES,
                                              resource)) {
            continue;
        }
        GLuint glUnit = resource ? resource->gl_binding
                                 : mglRendererGetProgramGLBinding(ctx, _TESS_EVALUATION_SHADER, _STORAGE_IMAGE_RES, (int)i);
        if (glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = MGL_STATE(glm_ctx)->image_units[glUnit].tex;
        if (ptr && !ptr->mtl_data) {
            [self bindMTLTexture:ptr];
        }
    }

    MGLStageBindingCopyBackList stageCopyBacks = {0};
    MGLTessStageBufferBindingList stageBufferBindings = {0};
    if (![self prepareTessStageBufferBindings:&stageBufferBindings
                                         stage:_TESS_EVALUATION_SHADER
                                     copyBacks:&stageCopyBacks]) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    MGLRenderCppComputeExecutionPlan executionPlan = {0};
    NSMutableArray *executionTemporaries = [NSMutableArray array];
    MGLMetalComputeCommandEncoderRef computeEncoder = nil;
    executionPlan.pipeline = (__bridge void *)tesPipeline;
    MGLMetalBufferRef patchInputs = (__bridge MGLMetalBufferRef)
        mglRendererBackendGetTcsPatchOutBuffer(_backend);
    if (!mglTessPlanBufferOrBind(
            &executionPlan,
            executionTemporaries, computeEncoder,
            _tessellation.tessFactorBuffer, 0u,
            MGL_AIR_TESS_SLOT_TESS_FACTOR) ||
        !mglTessPlanBufferOrBind(
            &executionPlan,
            executionTemporaries, computeEncoder,
            patchInputs ? patchInputs : outBuffer, 0u,
            MGL_AIR_TESS_SLOT_PATCH_OUT) ||
        !mglTessPlanBufferOrBind(
            &executionPlan,
            executionTemporaries, computeEncoder, outBuffer, 0u,
            MGL_AIR_TESS_SLOT_TCS_OUTPUT)) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    /* PASS 2: bind storage images for the TES stage. */
    for (GLuint i = 0; i < tesImgCount; i++) {
        MGLShaderResource *resource = NULL;
        if (i <
            tesProgram->shader_resources_list[_TESS_EVALUATION_SHADER][_STORAGE_IMAGE_RES].count) {
            resource =
                &tesProgram->shader_resources_list[_TESS_EVALUATION_SHADER][_STORAGE_IMAGE_RES].list[i];
        }
        if (mglShouldSkipStageTextureResource(tesProgram,
                                              _TESS_EVALUATION_SHADER,
                                              _STORAGE_IMAGE_RES,
                                              resource)) {
            continue;
        }
        GLuint metalSlot = resource ? mglMetalResourceSlot(resource)
                                    : mglRendererGetProgramBinding(ctx, _TESS_EVALUATION_SHADER, _STORAGE_IMAGE_RES, (int)i);
        GLuint glUnit = resource ? resource->gl_binding
                                 : mglRendererGetProgramGLBinding(ctx, _TESS_EVALUATION_SHADER, _STORAGE_IMAGE_RES, (int)i);
        if (metalSlot >= TEXTURE_UNITS || glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = MGL_STATE(glm_ctx)->image_units[glUnit].tex;
        if (ptr && !ptr->mtl_data) {
            [self bindMTLTexture:ptr];
        }
        MGLMetalTextureRef texture = ptr ? (__bridge MGLMetalTextureRef)(ptr->mtl_data) : nil;
        if (!mglTessPlanTextureOrBind(
                &executionPlan,
                executionTemporaries, computeEncoder, texture, metalSlot)) {
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }
    }

    /* Bind sampled textures + combined samplers for the TES stage. */
    GLuint tesSampledCount = mglRendererGetProgramBindingCount(ctx, _TESS_EVALUATION_SHADER, _SAMPLED_IMAGE_RES);
    for (GLuint i = 0; i < tesSampledCount; i++) {
        MGLShaderResource *resource = NULL;
        if (i <
            tesProgram->shader_resources_list[_TESS_EVALUATION_SHADER][_SAMPLED_IMAGE_RES].count) {
            resource =
                &tesProgram->shader_resources_list[_TESS_EVALUATION_SHADER][_SAMPLED_IMAGE_RES].list[i];
        }
        if (mglShouldSkipStageTextureResource(tesProgram,
                                              _TESS_EVALUATION_SHADER,
                                              _SAMPLED_IMAGE_RES,
                                              resource)) {
            continue;
        }
        GLuint metalSlot = resource ? mglMetalResourceSlot(resource)
                                    : mglRendererGetProgramBinding(ctx, _TESS_EVALUATION_SHADER, _SAMPLED_IMAGE_RES, (int)i);
        GLuint glUnit = resource ? resource->gl_binding
                                 : mglRendererGetProgramGLBinding(ctx, _TESS_EVALUATION_SHADER, _SAMPLED_IMAGE_RES, (int)i);
        if (metalSlot >= TEXTURE_UNITS || glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = MGL_STATE(glm_ctx)->active_textures[glUnit];
        if (ptr && !ptr->mtl_data) {
            [self bindMTLTexture:ptr];
        }
        MGLMetalTextureRef texture = ptr ? (__bridge MGLMetalTextureRef)(ptr->mtl_data) : nil;
        if (!mglTessPlanTextureOrBind(
                &executionPlan,
                executionTemporaries, computeEncoder, texture, metalSlot)) {
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }
        if (resource && resource->has_combined_sampler) {
            MGLMetalSamplerStateRef sampler = nil;
            if (MGL_STATE(glm_ctx)->texture_samplers[glUnit]) {
                Sampler *glSampler =
                    MGL_STATE(glm_ctx)->texture_samplers[glUnit];
                if (glSampler->dirty_bits && glSampler->mtl_data) {
                    mglSafeReleaseMetalObj((void **)&glSampler->mtl_data);
                }
                if (!glSampler->mtl_data && ptr) {
                    glSampler->mtl_data = (void *)CFBridgingRetain(
                        [self createMTLSamplerForTexParam:&glSampler->params
                                                  target:ptr->target]);
                    glSampler->dirty_bits = 0;
                }
                sampler = (__bridge MGLMetalSamplerStateRef)(glSampler->mtl_data);
            } else if (ptr && ptr->params.mtl_data) {
                sampler = (__bridge MGLMetalSamplerStateRef)(ptr->params.mtl_data);
            }
            if (!sampler) {
                sampler = mglTessCreateSampler(
                    _device, [MTLSamplerDescriptor new]);
            }
            if (sampler) {
                if (!mglTessPlanSamplerOrBind(
                        &executionPlan,
                        executionTemporaries, computeEncoder, sampler,
                        mglMetalCombinedSamplerSlot(resource))) {
                    [self clearStageBindingCopyBacks:&stageCopyBacks];
                    return false;
                }
            }
        }
    }

    if (![self bindPreparedTessStageBufferBindings:&stageBufferBindings
                                  toComputeEncoder:computeEncoder
                                     executionPlan:&executionPlan
                                      temporaries:executionTemporaries]) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    [self bindPointSizeParamsToComputeEncoder:computeEncoder
                                      program:tesProgram
                                        stage:_TESS_EVALUATION_SHADER
                                executionPlan:&executionPlan
                                 temporaries:executionTemporaries];

    /* Fixed ABI buffers (backend mgl_air_tess_abi.h §3): stage_in(24) is
     * the control point stream, factors(26) the quad-half records, patch
     * inputs(27) the TCS per-patch output, stageOut(28) the expanded
     * vertex records, indirect(29) the per-dispatch contract.  The stage_in
     * offset is rebased per instance (TES-only captures lay out
     * [instance][vertex]); the remaining ABI buffers are instance
     * invariant. */
    /* Transform-feedback stream (slot 31): the kernel writes complete stage
     * records. The renderer gathers selected varyings into the compact GL XFB
     * layout and copies only the prefix containing complete primitives. */
    TransformFeedback *xfbState = MGL_STATE(glm_ctx)->transform_feedback;
    const bool xfbActive =
        xfbState && xfbState->active && !xfbState->paused;
    MGLMetalBufferRef xfbTemporary = nil;
    MGLMetalBufferRef xfbCopyDestination = nil;
    Buffer *xfbDestination = NULL;
    NSUInteger xfbCopyDestinationOffset = 0u;
    NSUInteger xfbCompactStride = 0u;
    NSUInteger xfbCopiedVertices = 0u;
    NSUInteger xfbWrittenBytes = 0u;
    if (xfbActive) {
        BufferBaseTarget *xfbSlot =
            &MGL_STATE(glm_ctx)->buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[0];
        const GLenum xfbGenMode = tesProgram->tess_gen_mode;
        const GLboolean xfbPointMode = tesProgram->tess_gen_point_mode;
        NSUInteger captureVertices = 0u;
        NSUInteger requiredBytes = 0u;
        const bool sessionOffsetOK =
            xfbState->buffer_write_offsets[0] <= (GLuint64)NSUIntegerMax;
        const NSUInteger xfbSessionOffset =
            sessionOffsetOK ? (NSUInteger)xfbState->buffer_write_offsets[0] : 0u;
        xfbCompactStride = mglTESXFBVertexStride(tesProgram);
        const bool sizeOK = xfbCompactStride > 0u &&
            mglCheckedNSUIntegerProduct((NSUInteger)itemsPerInstanceU,
                                        (NSUInteger)instanceCountU,
                                        &captureVertices) &&
            mglCheckedNSUIntegerProduct(captureVertices, outStride,
                                       &requiredBytes) &&
            requiredBytes > 0u;

        MGLMetalBufferRef xfbMTL = nil;
        NSUInteger visibleBytes = 0u;
        NSUInteger remainingVisibleBytes = 0u;
        NSUInteger destinationOffset = 0u;
        bool destinationOffsetOK = false;
        if (xfbSlot->buf) {
            if (xfbSlot->buf->data.dirty_bits &
                (DIRTY_BUFFER_DATA | DIRTY_BUFFER_ADDR)) {
                /* Consume CPU initialization before the XFB blit writes the
                 * same backing. Otherwise a later map can upload the stale
                 * shadow over the captured GPU data. */
                if (![self updateDirtyBuffer:xfbSlot->buf]) {
                    [self clearStageBindingCopyBacks:&stageCopyBacks];
                    return false;
                }
            }
            if (!xfbSlot->buf->data.mtl_data) {
                [self bindMTLBuffer:xfbSlot->buf];
            }
            xfbMTL = (__bridge MGLMetalBufferRef)(xfbSlot->buf->data.mtl_data);
            if (xfbMTL) {
                BufferMap xfbMap = {0};
                xfbMap.buf = xfbSlot->buf;
                xfbMap.offset = xfbSlot->offset;
                xfbMap.size = xfbSlot->size;
                visibleBytes =
                    mglBufferMapVisibleBackingBytes(&xfbMap, (size_t)xfbMTL.length);
                if (xfbSessionOffset <= visibleBytes &&
                    xfbSlot->offset >= 0 &&
                    (NSUInteger)xfbSlot->offset <=
                        NSUIntegerMax - xfbSessionOffset) {
                    remainingVisibleBytes = visibleBytes - xfbSessionOffset;
                    destinationOffset =
                        (NSUInteger)xfbSlot->offset + xfbSessionOffset;
                    destinationOffsetOK = true;
                }
            }
        }

        if (sizeOK) {
            const GLuint verticesPerPrimitive =
                xfbPointMode ? 1u : (xfbGenMode == GL_ISOLINES ? 2u : 3u);
            NSUInteger primitiveBytes = 0u;
            const bool primitiveLayoutOK =
                mglCheckedNSUIntegerProduct((NSUInteger)verticesPerPrimitive,
                                            xfbCompactStride, &primitiveBytes) &&
                primitiveBytes > 0u;
            const NSUInteger capturePrimitives = primitiveLayoutOK
                ? captureVertices / (NSUInteger)verticesPerPrimitive : 0u;
            /* The AIR kernel writes full stage records (built-ins followed by
             * location-based user outputs). GL XFB is a compact stream of only
             * the selected varyings, so it can never target the GL range
             * directly. Gather the selected fields after the dispatch. */
            xfbTemporary = mglTessCreateBuffer(
                _device, requiredBytes, MTLResourceStorageModeShared);
            if (!xfbTemporary) {
                [self clearStageBindingCopyBacks:&stageCopyBacks];
                return false;
            }
            if (!mglTessPlanBufferOrBind(
                    &executionPlan,
                    executionTemporaries, computeEncoder, xfbTemporary, 0u,
                    MGL_AIR_TESS_SLOT_XFB_OUT)) {
                [self clearStageBindingCopyBacks:&stageCopyBacks];
                return false;
            }
            if (primitiveLayoutOK && xfbMTL && destinationOffsetOK) {
                NSUInteger copiedPrimitives = MIN(
                    capturePrimitives, remainingVisibleBytes / primitiveBytes);
                xfbCopiedVertices =
                    copiedPrimitives * (NSUInteger)verticesPerPrimitive;
                xfbWrittenBytes = copiedPrimitives * primitiveBytes;
                xfbCopyDestination = xfbMTL;
                xfbCopyDestinationOffset = destinationOffset;
                xfbDestination = xfbSlot->buf;
            }
        }
    } else {
        /* The TES compute kernel always declares and writes the XFB stream
         * slot (31); bind a 1-byte dummy so the slot is never dangling when
         * GL feedback is inactive. */
        void *cachedDummy = NULL;
        MGLMetalBufferRef xfbDummy = nil;
        if (mglRendererBackendGetTessXfbDummyBuffer(
                _backend, MAX(outSize, 1u), &cachedDummy) == 1) {
            xfbDummy = (__bridge MGLMetalBufferRef)cachedDummy;
        }
        if (!xfbDummy) {
            xfbDummy = mglTessCreateBuffer(
                _device, MAX(outSize, 1u), MTLResourceStorageModeShared);
            if (xfbDummy) {
                (void)mglRendererBackendPutTessXfbDummyBuffer(
                    _backend, (__bridge void *)xfbDummy);
            }
        }
        if (xfbDummy) {
            if (!mglTessPlanBufferOrBind(
                    &executionPlan,
                    executionTemporaries, computeEncoder,
                    xfbDummy, 0u,
                    MGL_AIR_TESS_SLOT_XFB_OUT)) {
                [self clearStageBindingCopyBacks:&stageCopyBacks];
                return false;
            }
        }
    }

    /* Dispatch per patch (item counts differ per patch).  The contract
     * {patch_id, gl_in_vertices, items, output_item_base} is written for
     * each dispatch; output_item_base spans instances first so each
     * instance owns a contiguous [instance*itemsPerInstance] span. */
    uint32_t *patchBases = (uint32_t *)malloc(
        (size_t)(patchCount + 1u) * sizeof(uint32_t));
    if (!patchBases) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    {
        uint32_t base = 0u;
        for (GLuint p = 0u; p < patchCount; p++) {
            patchBases[p] = base;
            const void *record =
                (const void *)((const uint8_t *)factorBytes +
                               (NSUInteger)p *
                                   MGL_AIR_TESS_FACTOR_QUAD_HALF_BYTES);
            GLuint items = mglAIRTessEvalItemsPerPatch(tesProgram, record);
            if (items == 0u) items = 1u;
            base += items;
        }
        patchBases[patchCount] = base;
    }
    uint32_t contractWords[4];
    contractWords[1] = glInVertices;
    const BOOL indexed = _tessellation.tessIndexedDraw;
    MGLMetalBufferRef gatherBuffer = indexed
        ? controlPointIndexBuffer : nil;
    const GLuint gatherFirstVertex = 0u;
    const GLuint gatherVertsPerInstance =
        indexed ? (GLuint)_tessellation.tessInstanceRecords
                : MAX(1u, contract->patch_vertices);
    const GLuint gatherPrimsPerInstance =
        indexed ? patchCount : 0u;
    for (GLuint inst = 0u; inst < instanceCountU; inst++) {
        const NSUInteger instGlInOffset =
            indexed ? 0u
                    : glInOffset + (NSUInteger)inst * glInInstanceStride;
        if (!mglTessPlanBufferOrBind(
                &executionPlan,
                executionTemporaries, computeEncoder,
                glInBuffer, instGlInOffset,
                MGL_AIR_TESS_SLOT_TCS_STAGE_IN)) {
            free(patchBases);
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }
        if (gatherBuffer) {
            if (!mglTessPlanBufferOrBind(
                    &executionPlan,
                    executionTemporaries, computeEncoder, gatherBuffer, 0u,
                    MGL_AIR_TESS_SLOT_GATHER_INDEX)) {
                free(patchBases);
                [self clearStageBindingCopyBacks:&stageCopyBacks];
                return false;
            }
        }
        {
            const GLuint gatherParams[5] = {
                gatherVertsPerInstance, gatherPrimsPerInstance,
                gatherFirstVertex, indexed ? 1u : 0u, inst,
            };
            if (!mglTessPlanBytesOrBind(
                    &executionPlan,
                    executionTemporaries, computeEncoder, gatherParams,
                    sizeof(gatherParams), MGL_AIR_TESS_SLOT_GATHER_PARAMS)) {
                free(patchBases);
                [self clearStageBindingCopyBacks:&stageCopyBacks];
                return false;
            }
        }
        for (GLuint p = 0u; p < patchCount; p++) {
            const void *record =
                (const void *)((const uint8_t *)factorBytes +
                               (NSUInteger)p *
                                   MGL_AIR_TESS_FACTOR_QUAD_HALF_BYTES);
            GLuint items = mglAIRTessEvalItemsPerPatch(tesProgram, record);
            if (items == 0u) items = 1u;
            contractWords[0] = p;
            contractWords[2] = items;
            contractWords[3] = inst * itemsPerInstanceU + patchBases[p];
            if (!mglTessPlanBytesOrBind(
                    &executionPlan,
                    executionTemporaries, computeEncoder, contractWords,
                    sizeof(contractWords), MGL_AIR_TESS_SLOT_INDIRECT) ||
                !mglTessPlanDispatchOrBind(
                    &executionPlan, computeEncoder,
                    MTLSizeMake((items + 63u) / 64u, 1u, 1u),
                    MTLSizeMake(64u, 1u, 1u))) {
                free(patchBases);
                [self clearStageBindingCopyBacks:&stageCopyBacks];
                return false;
            }
        }
    }
    free(patchBases);
    {
        MGLRenderCppCopyBackEntry copyBackEntries[kMGLMaxBufferSlots] = {0};
        uint32_t copyBackEntryCount = 0u;
        for (NSUInteger slot = 0; slot < kMGLMaxBufferSlots; slot++) {
            MGLStageBindingCopyBack *entry = &stageCopyBacks.slots[slot];
            if (entry->length == 0u) continue;
            copyBackEntries[copyBackEntryCount++] =
                (MGLRenderCppCopyBackEntry){
                    .temporary = (__bridge void *)entry->temporary,
                    .destination = (__bridge void *)entry->destination,
                    .destination_buffer = entry->destination_buffer,
                    .destination_offset = entry->destination_offset,
                    .length = entry->length,
                };
        }
        executionPlan.barrier_scope = MGL_RENDER_CPP_COMPUTE_BARRIER_BUFFERS;
        MGLRenderCppComputeExecutionResult executionResult = {0};
        char executionError[256] = {0};
        if (mglRenderCppExecuteComputeExecutionPlan(
                _renderPassManager.state->currentCommandBufferOwner,
                _gpuRecovery.commandRecoveryOwner,
                &executionPlan, copyBackEntries, copyBackEntryCount, 1u,
                &executionResult, executionError,
                sizeof(executionError)) != 0) {
            if (executionResult.transaction.device_reset_requested) {
                atomic_store_explicit(&_deviceResetRequested, true,
                                      memory_order_release);
            }
            NSLog(@"MGL TESS ERROR: C++ TES execution failed: %s",
                  executionError[0] ? executionError : "unknown error");
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }
        [self clearStageBindingCopyBacks:&stageCopyBacks];
    }

    if (xfbWrittenBytes > 0u && xfbTemporary && xfbCopyDestination) {
        const MGLShaderResourceList *outputs =
            &tesProgram->shader_resources_list[_TESS_EVALUATION_SHADER]
                                                [_STAGE_OUTPUT_RES];
        const NSUInteger varyingCount =
            (NSUInteger)MAX(tesProgram->transform_feedback_varying_count, 0);
        NSUInteger copyCapacity = 0u;
        if (!mglCheckedNSUIntegerProduct(xfbCopiedVertices, varyingCount,
                                         &copyCapacity) ||
            copyCapacity > UINT32_MAX ||
            copyCapacity > SIZE_MAX / sizeof(MGLRenderCppBufferCopyEntry)) {
            NSLog(@"MGL TESS XFB: copy plan size overflow");
            return false;
        }
        MGLRenderCppBufferCopyEntry *xfbCopies = copyCapacity
            ? (MGLRenderCppBufferCopyEntry *)calloc(
                  copyCapacity, sizeof(MGLRenderCppBufferCopyEntry))
            : NULL;
        if (copyCapacity && !xfbCopies) {
            NSLog(@"MGL TESS XFB: copy plan allocation failed");
            return false;
        }
        uint32_t xfbCopyCount = 0u;
        for (NSUInteger vertex = 0u; vertex < xfbCopiedVertices; vertex++) {
            NSUInteger compactOffset = 0u;
            for (GLsizei varying = 0;
                 varying < tesProgram->transform_feedback_varying_count;
                 varying++) {
                const char *name =
                    tesProgram->transform_feedback_varying_names[varying];
                const MGLShaderResource *output = NULL;
                for (GLuint i = 0u; name && outputs->list && i < outputs->count;
                     i++) {
                    if (outputs->list[i].name &&
                        strcmp(outputs->list[i].name, name) == 0) {
                        output = &outputs->list[i];
                        break;
                    }
                }
                NSUInteger fieldBytes =
                    output ? mglTESXFBFieldByteSize(output->gl_type) : 0u;
                if (!output || fieldBytes == 0u) {
                    continue;
                }
                NSUInteger sourceOffset = vertex * outStride +
                    MGL_AIR_PER_VERTEX_STRIDE +
                    (NSUInteger)output->location * 16u;
                NSUInteger destinationOffset = xfbCopyDestinationOffset +
                    vertex * xfbCompactStride + compactOffset;
                xfbCopies[xfbCopyCount++] = (MGLRenderCppBufferCopyEntry){
                    .source_buffer = (__bridge void *)xfbTemporary,
                    .source_offset = sourceOffset,
                    .destination_buffer = (__bridge void *)xfbCopyDestination,
                    .destination_offset = destinationOffset,
                    .length = fieldBytes,
                };
                compactOffset += fieldBytes;
            }
        }
        const bool xfbCopyOK = xfbCopyCount == 0u ||
            mglTessEncodeBufferCopiesForOwner(
                _renderPassManager.state->currentCommandBufferOwner,
                xfbCopies, xfbCopyCount);
        free(xfbCopies);
        if (!xfbCopyOK) {
            NSLog(@"MGL TESS XFB: failed to encode bounded copies");
            return false;
        }
        if (xfbDestination) {
            xfbDestination->ever_written = GL_TRUE;
        }
    }
    if (xfbActive && xfbWrittenBytes > 0u) {
        const GLuint64 currentOffset = xfbState->buffer_write_offsets[0];
        xfbState->buffer_write_offsets[0] =
            (GLuint64)xfbWrittenBytes > UINT64_MAX - currentOffset
                ? UINT64_MAX
                : currentOffset + (GLuint64)xfbWrittenBytes;
    }

    /* Rasterize through the passthrough vertex stage. */
    const GLenum genMode = tesProgram->tess_gen_mode;
    const GLboolean pointMode = tesProgram->tess_gen_point_mode;
    const uint64_t primitivesPerInstance =
        pointMode ? itemsPerInstanceU
                  : (genMode == GL_ISOLINES ? itemsPerInstanceU / 2u
                                            : itemsPerInstanceU / 3u);
    if (MGL_STATE(glm_ctx)->caps.rasterizer_discard) {
        /* GL_RASTERIZER_DISCARD: no pixels by definition, so skip the
         * passthrough draw entirely, but the compute expansion already ran
         * and the primitive query must still count the generated
         * primitives (persistent query semantics). */
        GLuint64 prims = (GLuint64)instanceCount * primitivesPerInstance;
        GLuint64 written = prims;
        if (xfbActive) {
            const GLuint64 vpp = pointMode ? 1u
                : (genMode == GL_ISOLINES ? 2u : 3u);
            const GLuint64 xfbPrims =
                xfbWrittenBytes / ((GLuint64)xfbCompactStride * vpp);
            written = MIN(written, xfbPrims);
        }
        _currentCBHasWork = YES;
        mglRecordActivePrimitiveQueryDraw(glm_ctx, prims, written);
        return YES;
    }
    if (![self ensureAIRTessEvalPassthroughFunctionForProgram:tesProgram]) {
        NSLog(@"MGL TESS ERROR: TES passthrough vertex unavailable program=%u",
              (unsigned)tesProgram->name);
        return false;
    }
    MTLPrimitiveType primType = pointMode ? MTLPrimitiveTypePoint
        : (genMode == GL_ISOLINES ? MTLPrimitiveTypeLine
                                  : MTLPrimitiveTypePoint);

    _tessellation.tessComputeActive = YES;
    _tessellation.tessComputeProgram = tesProgram;
    BOOL stateReady = [self processGLState:true];
    if (!stateReady ||
        mglRenderCppRenderEncoderOwnerHasCurrent(
            _renderPassManager.state->currentRenderEncoderOwner) != 1 ||
        [self currentDrawRasterizationIsEmpty]) {
        _tessellation.tessComputeActive = NO;
        _tessellation.tessComputeProgram = NULL;
        if (xfbActive) {
            const GLuint64 vpp = pointMode ? 1u
                : (genMode == GL_ISOLINES ? 2u : 3u);
            GLuint64 prims = (GLuint64)instanceCount * primitivesPerInstance;
            GLuint64 written = MIN(prims,
                xfbWrittenBytes / ((GLuint64)xfbCompactStride * vpp));
            mglRecordActivePrimitiveQueryDraw(glm_ctx, prims, written);
        }
        return NO;
    }

    [self applyPolygonOffsetForDrawMode:genMode == GL_ISOLINES
                                            ? GL_LINES
                                            : GL_POINTS];
    MGLMetalRenderCommandEncoderRef encoder = nil;
    for (GLsizei i = 0; i < instanceCount; i++) {
        NSUInteger instanceOffset =
            (NSUInteger)i * (NSUInteger)itemsPerInstanceU * outStride;
        mglTessSetRenderVertexBuffer(
            encoder, _renderPassManager.state->currentRenderEncoderOwner,
            outBuffer, instanceOffset, 0u);
        mglTessDrawPrimitives(
            encoder, _renderPassManager.state->currentRenderEncoderOwner,
            primType, 0u, (NSUInteger)itemsPerInstanceU, 1u,
            (NSUInteger)baseInstance + (NSUInteger)i);
    }
    _currentCBHasWork = YES;
    GLuint64 prims = (GLuint64)instanceCount * primitivesPerInstance;
    GLuint64 written = prims;
    if (xfbActive) {
        const GLuint64 vpp = pointMode ? 1u
            : (genMode == GL_ISOLINES ? 2u : 3u);
        written = MIN(written, xfbWrittenBytes /
                               ((GLuint64)xfbCompactStride * vpp));
    }
    mglRecordActivePrimitiveQueryDraw(glm_ctx, prims, written);
    _tessellation.tessComputeActive = NO;
    _tessellation.tessComputeProgram = NULL;
    return YES;
}

/* Dispatch a TES (Tessellation Evaluation Shader) when there is no TCS and
 * GL_RASTERIZER_DISCARD is active.  The AIR backend lowers the TES to a Metal
 * post-tessellation vertex function (`[[patch(quad, 0)]] vertex ...`), but
 * macOS 26.5 SDK removed postTessellationVertexFunction / isTessellationEnabled
 * from MTLRenderPipelineDescriptor.  We therefore rewrite the TES MSL to a
 * plain compute kernel (mglFixMSLTesAsComputeKernel in program.c) and dispatch
 * it with a compute pipeline, exactly like TCS.
 *
 * The TES kernel uses gl_PrimitiveID (mapped to threadgroup_position_in_grid)
 * as the patch index.  We dispatch one threadgroup per patch with 1 thread
 * per threadgroup, so each invocation handles one patch. */
-(bool) dispatchTessEvaluationShader:(GLMContext) glm_ctx
                            program:(Program *) tesProgram
                           contract:(const MGLAIRTessDrawContract *) contract
{
    if (!tesProgram || !glm_ctx || !contract) {
        return false;
    }

    Shader *tesShader = tesProgram->shader_slots[_TESS_EVALUATION_SHADER];
    if (!tesShader || !tesProgram->modules[_TESS_EVALUATION_SHADER].mtl_function) {
        NSLog(@"MGL TESS WARNING: TES program %u has no compiled function", tesProgram->name);
        return false;
    }

    /* Create compute pipeline state for TES kernel. */
    void *tesPipelineHandle = NULL;
    char tesPipelineError[512] = {0};
    int tesPipelineResult = mglGetOrCreateProgramComputePipeline(
        tesProgram, _TESS_EVALUATION_SHADER, &tesPipelineHandle,
        tesPipelineError, sizeof(tesPipelineError));
    MGLMetalComputePipelineStateRef tesPipeline =
        tesPipelineResult == 0 && tesPipelineHandle
            ? (__bridge_transfer MGLMetalComputePipelineStateRef)tesPipelineHandle
            : nil;
    if (!tesPipeline) {
        NSLog(@"MGL TESS ERROR: failed to create TES compute pipeline for program %u: %s",
              tesProgram->name,
              tesPipelineError[0] ? tesPipelineError : "unknown error");
        return false;
    }

    /* PASS 1: Pre-resolve all Metal textures that the TES kernel needs.
     * Must happen before opening any encoder (same reason as TCS). */
    if (mglRenderCppRenderEncoderOwnerHasCurrent(
            _renderPassManager.state->currentRenderEncoderOwner) == 1) {
        [self endRenderEncoding];
    }

    /* Ensure a writable command buffer exists (same reason as TCS). */
    MGLRenderCppCommandBufferState commandState = {0};
    if (!mglRenderCommandBufferOwnerState(
            _renderPassManager.state->currentCommandBufferOwner,
            &commandState) ||
        commandState.status >= MTLCommandBufferStatusCommitted) {
        if (![self newCommandBuffer]) {
            NSLog(@"MGL TESS ERROR: failed to create command buffer for TES dispatch");
            return false;
        }
    }

    GLuint tesImgCount = mglRendererGetProgramBindingCount(ctx, _TESS_EVALUATION_SHADER, _STORAGE_IMAGE_RES);
    for (GLuint i = 0; i < tesImgCount; i++) {
        MGLShaderResource *resource = NULL;
        if (tesProgram &&
            i < tesProgram->shader_resources_list[_TESS_EVALUATION_SHADER][_STORAGE_IMAGE_RES].count) {
            resource = &tesProgram->shader_resources_list[_TESS_EVALUATION_SHADER][_STORAGE_IMAGE_RES].list[i];
        }
        if (mglShouldSkipStageTextureResource(tesProgram,
                                              _TESS_EVALUATION_SHADER,
                                              _STORAGE_IMAGE_RES,
                                              resource)) {
            continue;
        }
        GLuint glUnit = resource ? resource->gl_binding
                                 : mglRendererGetProgramGLBinding(ctx, _TESS_EVALUATION_SHADER, _STORAGE_IMAGE_RES, (int)i);
        if (glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = MGL_STATE(ctx)->image_units[glUnit].tex;
        if (ptr && !ptr->mtl_data) {
            [self bindMTLTexture:ptr];
        }
    }

    MGLStageBindingCopyBackList stageCopyBacks = {0};
    MGLTessStageBufferBindingList stageBufferBindings = {0};
    if (![self prepareTessStageBufferBindings:&stageBufferBindings
                                         stage:_TESS_EVALUATION_SHADER
                                     copyBacks:&stageCopyBacks]) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    MGLRenderCppComputeExecutionPlan executionPlan = {0};
    NSMutableArray *executionTemporaries = [NSMutableArray array];
    MGLMetalComputeCommandEncoderRef computeEncoder = nil;
    executionPlan.pipeline = (__bridge void *)tesPipeline;

    /* PASS 2: Bind storage images for TES stage. */
    for (GLuint i = 0; i < tesImgCount; i++) {
        MGLShaderResource *resource = NULL;
        if (tesProgram &&
            i < tesProgram->shader_resources_list[_TESS_EVALUATION_SHADER][_STORAGE_IMAGE_RES].count) {
            resource = &tesProgram->shader_resources_list[_TESS_EVALUATION_SHADER][_STORAGE_IMAGE_RES].list[i];
        }
        if (mglShouldSkipStageTextureResource(tesProgram,
                                              _TESS_EVALUATION_SHADER,
                                              _STORAGE_IMAGE_RES,
                                              resource)) {
            continue;
        }
        GLuint metalSlot = resource ? mglMetalResourceSlot(resource)
                                    : mglRendererGetProgramBinding(ctx, _TESS_EVALUATION_SHADER, _STORAGE_IMAGE_RES, (int)i);
        GLuint glUnit = resource ? resource->gl_binding
                                 : mglRendererGetProgramGLBinding(ctx, _TESS_EVALUATION_SHADER, _STORAGE_IMAGE_RES, (int)i);
        if (metalSlot >= TEXTURE_UNITS || glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = MGL_STATE(ctx)->image_units[glUnit].tex;
        MGLMetalTextureRef texture = nil;
        if (ptr) {
            texture = (__bridge MGLMetalTextureRef)(ptr->mtl_data);
            GLuint imgLevel = MGL_STATE(ctx)->image_units[glUnit].level;
            if (imgLevel > 0u && texture) {
                NSUInteger sliceCount = texture.arrayLength;
                if (texture.textureType == MTLTextureTypeCube ||
                    texture.textureType == MTLTextureTypeCubeArray) {
                    sliceCount = texture.arrayLength * 6u;
                }
                MGLMetalTextureRef levelView =
                    mglTessCreateTextureLevelView(
                        texture, imgLevel, sliceCount);
                if (levelView) {
                    texture = levelView;
                }
            }
        }
        if (!mglTessPlanTextureOrBind(
                &executionPlan,
                executionTemporaries, computeEncoder, texture, metalSlot)) {
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }
    }

    /* Also bind sampled (read-only) images for TES stage. */
    GLuint tesSampledCount = mglRendererGetProgramBindingCount(ctx, _TESS_EVALUATION_SHADER, _SAMPLED_IMAGE_RES);
    for (GLuint i = 0; i < tesSampledCount; i++) {
        MGLShaderResource *resource = NULL;
        if (tesProgram &&
            i < tesProgram->shader_resources_list[_TESS_EVALUATION_SHADER][_SAMPLED_IMAGE_RES].count) {
            resource = &tesProgram->shader_resources_list[_TESS_EVALUATION_SHADER][_SAMPLED_IMAGE_RES].list[i];
        }
        if (mglShouldSkipStageTextureResource(tesProgram,
                                              _TESS_EVALUATION_SHADER,
                                              _SAMPLED_IMAGE_RES,
                                              resource)) {
            continue;
        }
        GLuint metalSlot = resource ? mglMetalResourceSlot(resource)
                                    : mglRendererGetProgramBinding(ctx, _TESS_EVALUATION_SHADER, _SAMPLED_IMAGE_RES, (int)i);
        GLuint glUnit = resource ? resource->gl_binding
                                 : mglRendererGetProgramGLBinding(ctx, _TESS_EVALUATION_SHADER, _SAMPLED_IMAGE_RES, (int)i);
        if (metalSlot >= TEXTURE_UNITS || glUnit >= TEXTURE_UNITS) {
            continue;
        }
        Texture *ptr = MGL_STATE(ctx)->active_textures[glUnit];
        if (ptr && !ptr->mtl_data) {
            [self bindMTLTexture:ptr];
        }
        MGLMetalTextureRef texture = ptr ? (__bridge MGLMetalTextureRef)(ptr->mtl_data) : nil;
        if (!mglTessPlanTextureOrBind(
                &executionPlan,
                executionTemporaries, computeEncoder, texture, metalSlot)) {
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }
        if (resource && resource->has_combined_sampler) {
            MGLMetalSamplerStateRef sampler = nil;
            if (MGL_STATE(ctx)->texture_samplers[glUnit]) {
                Sampler *glSampler = MGL_STATE(ctx)->texture_samplers[glUnit];
                if (glSampler->dirty_bits && glSampler->mtl_data) {
                    mglSafeReleaseMetalObj((void **)&glSampler->mtl_data);
                }
                if (!glSampler->mtl_data && ptr) {
                    glSampler->mtl_data = (void *)CFBridgingRetain(
                        [self createMTLSamplerForTexParam:&glSampler->params
                                                  target:ptr->target]);
                    glSampler->dirty_bits = 0;
                }
                sampler = (__bridge MGLMetalSamplerStateRef)(glSampler->mtl_data);
            } else if (ptr && ptr->params.mtl_data) {
                sampler = (__bridge MGLMetalSamplerStateRef)(ptr->params.mtl_data);
            }
            if (!sampler) {
                sampler = mglTessCreateSampler(
                    _device, [MTLSamplerDescriptor new]);
            }
            if (sampler) {
                if (!mglTessPlanSamplerOrBind(
                        &executionPlan,
                        executionTemporaries, computeEncoder, sampler,
                        mglMetalCombinedSamplerSlot(resource))) {
                    [self clearStageBindingCopyBacks:&stageCopyBacks];
                    return false;
                }
            }
        }
    }

    /* Bind stage buffers (UBO, SSBO, atomic counters) for TES. */
    if (![self bindPreparedTessStageBufferBindings:&stageBufferBindings
                                  toComputeEncoder:computeEncoder
                                     executionPlan:&executionPlan
                                      temporaries:executionTemporaries]) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    [self bindPointSizeParamsToComputeEncoder:computeEncoder
                                      program:tesProgram
                                        stage:_TESS_EVALUATION_SHADER
                                executionPlan:&executionPlan
                                 temporaries:executionTemporaries];

    /* Dispatch: one threadgroup per patch, 1 thread per threadgroup.
     * gl_PrimitiveID → threadgroup_position_in_grid gives the patch index.
     * TessCoord → thread_position_in_threadgroup is 0 (1 thread per TG). */
    const GLuint patchVertices = MAX(1u, contract->patch_vertices);
    const GLuint patchCount = MAX(1u, contract->patch_count);

    /* Bind patch info to buffer(28): {patch_vertices_in, tcs_out_vertices}.
     * _mgl_patch_info.x = patch vertices (gl_in.size() replacement)
     * _mgl_patch_info.y = TCS output vertices per patch (for per-patch gl_in indexing) */
    {
        GLuint patchInfo[2] = { patchVertices, _tessellation.tcsOutVertices };
        if (patchInfo[1] == 0) patchInfo[1] = patchVertices;
        if (!mglTessPlanBytesOrBind(
                &executionPlan,
                executionTemporaries, computeEncoder, patchInfo,
                sizeof(patchInfo), MGL_AIR_TESS_SLOT_PATCH_INFO)) {
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }
    }

    /* Bind TCS output buffer to buffer(30) for TES gl_in.
     * TCS writes per-vertex output to spvOut (buffer 28 in TCS).  TES reads
     * gl_in[...] from buffer(30).  The data layout is: TCS writes
     * spvOut[patchID * outputVertices + invocationID], so TES gl_in should
     * point to the same buffer.  The MSL rewriter changed TES's [[stage_in]]
     * to "device <type> *gl_in [[buffer(30)]]". */
    if (_tessellation.tcsOutputBuffer) {
        if (!mglTessPlanBufferOrBind(
                &executionPlan,
                executionTemporaries, computeEncoder,
                _tessellation.tcsOutputBuffer, 0,
                MGL_AIR_TESS_SLOT_GL_IN)) {
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }
    }

    /* Bind TCS patch output buffer to buffer(27) for TES patchIn.
     * TCS writes per-patch output to spvPatchOut (buffer 27 in TCS).  TES
     * reads patchIn[...] from buffer(27).  Note: buffer 27 is reused for both
     * TCS spvPatchOut and TES patchIn, which is correct since the data flows
     * TCS → TES. */
    MGLMetalBufferRef tcsPatchOutBuffer = (__bridge MGLMetalBufferRef)
        mglRendererBackendGetTcsPatchOutBuffer(_backend);
    if (tcsPatchOutBuffer) {
        if (!mglTessPlanBufferOrBind(
                &executionPlan,
                executionTemporaries, computeEncoder,
                tcsPatchOutBuffer, 0,
                MGL_AIR_TESS_SLOT_PATCH_OUT)) {
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }
    }

    /* Compute vertsPerPatch from tessellation factors.
     * We dispatch vertsPerPatch threads per threadgroup so each thread
     * writes one XFB entry.  The vertex count formula matches what the
     * CTS counter program expects (primitive count * vertices-per-primitive). */
    GLuint vertsPerPatch = 1;
    if (_tessellation.tessFactorBuffer) {
        const struct {
            uint16_t edge[4];
            uint16_t inside[2];
        } __attribute__((packed)) *tf = (const void *)_tessellation.tessFactorBuffer.contents;
        GLenum genMode = tesProgram ? tesProgram->tess_gen_mode : GL_TRIANGLES;
        GLboolean pointMode = tesProgram ? tesProgram->tess_gen_point_mode : GL_FALSE;
        if (patchCount > 0) {
            float edge0 = *(const __fp16 *)&tf[0].edge[0];
            float inside0 = *(const __fp16 *)&tf[0].inside[0];
            float inside1 = *(const __fp16 *)&tf[0].inside[1];
            if (edge0 < 1.0f) edge0 = 1.0f;
            if (inside0 < 1.0f) inside0 = 1.0f;
            if (inside1 < 1.0f) inside1 = 1.0f;
            GLuint primPerPatch = 1;
            if (genMode == GL_QUADS) {
                primPerPatch = 2u * (GLuint)ceilf(inside0) * (GLuint)ceilf(inside1);
            } else if (genMode == GL_TRIANGLES) {
                primPerPatch = (GLuint)ceilf(inside0) * (GLuint)ceilf(inside0);
            } else { /* GL_ISOLINES */
                primPerPatch = (GLuint)ceilf(edge0);
            }
            if (primPerPatch == 0u) primPerPatch = 1u;
            if (pointMode) {
                vertsPerPatch = primPerPatch;
            } else if (genMode == GL_ISOLINES) {
                vertsPerPatch = primPerPatch * 2u;
            } else {
                vertsPerPatch = primPerPatch * 3u;
            }
        }
    }
    if (vertsPerPatch == 0) vertsPerPatch = 1;

    /* Bind XFB output buffer to buffer(29) for _mgl_xfb_out. Metal buffer
     * arguments cannot express a subrange, so a direct binding is safe only
     * when every injected write fits in both the requested GL range and the
     * current logical store. On overflow, capture into a full-size temporary
     * buffer and copy back only the prefix containing complete primitives. */
    TransformFeedback *xfbState = MGL_STATE(glm_ctx)->transform_feedback;
    const bool xfbCaptureActive = false;
    MGLMetalBufferRef xfbTemporary = nil;
    MGLMetalBufferRef xfbCopyDestination = nil;
    Buffer *xfbDestination = NULL;
    NSUInteger xfbCopyDestinationOffset = 0u;
    NSUInteger xfbCopyBytes = 0u;
    NSUInteger xfbPrimitiveCapacity = 0u;
    NSUInteger xfbWrittenBytes = 0u;

    if (xfbCaptureActive) {
        BufferBaseTarget *xfbSlot =
            &MGL_STATE(glm_ctx)->buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[0];
        NSUInteger xfbStride = mglTESXFBVertexStride(tesProgram);
        NSUInteger conservativeStride =
            (NSUInteger)tesProgram->transform_feedback_varying_count * 16u;
        NSUInteger allocationStride = xfbStride ? xfbStride : conservativeStride;
        NSUInteger captureVertices = 0u;
        NSUInteger requiredBytes = 0u;
        NSUInteger xfbSessionOffset = 0u;
        bool sessionOffsetOK =
            xfbState->buffer_write_offsets[0] <= (GLuint64)NSUIntegerMax;
        if (sessionOffsetOK) {
            xfbSessionOffset = (NSUInteger)xfbState->buffer_write_offsets[0];
        }
        bool sizeOK =
            allocationStride > 0u &&
            mglCheckedNSUIntegerProduct((NSUInteger)patchCount,
                                        (NSUInteger)vertsPerPatch,
                                        &captureVertices) &&
            mglCheckedNSUIntegerProduct(captureVertices,
                                        allocationStride,
                                        &requiredBytes) &&
            requiredBytes > 0u;

        MGLMetalBufferRef xfbMTL = nil;
        NSUInteger visibleBytes = 0u;
        NSUInteger remainingVisibleBytes = 0u;
        NSUInteger destinationOffset = 0u;
        bool destinationOffsetOK = false;
        if (xfbSlot->buf) {
            if (!xfbSlot->buf->data.mtl_data) {
                [self bindMTLBuffer:xfbSlot->buf];
            }
            xfbMTL = (__bridge MGLMetalBufferRef)(xfbSlot->buf->data.mtl_data);
            if (xfbMTL) {
                BufferMap xfbMap = {0};
                xfbMap.buf = xfbSlot->buf;
                xfbMap.offset = xfbSlot->offset;
                xfbMap.size = xfbSlot->size;
                visibleBytes = mglBufferMapVisibleBackingBytes(
                    &xfbMap, (size_t)xfbMTL.length);
                if (sessionOffsetOK && xfbSessionOffset <= visibleBytes &&
                    xfbSlot->offset >= 0 &&
                    (NSUInteger)xfbSlot->offset <= NSUIntegerMax - xfbSessionOffset) {
                    remainingVisibleBytes = visibleBytes - xfbSessionOffset;
                    destinationOffset = (NSUInteger)xfbSlot->offset + xfbSessionOffset;
                    destinationOffsetOK = true;
                }
            }
        }

        if (!sizeOK) {
            NSLog(@"MGL TESS XFB: capture size overflow for program %u",
                  (unsigned)tesProgram->name);
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }

        GLuint verticesPerPrimitive =
            tesProgram->tess_gen_point_mode ? 1u :
            (tesProgram->tess_gen_mode == GL_ISOLINES ? 2u : 3u);
        NSUInteger primitiveBytes = 0u;
        bool primitiveLayoutOK =
            xfbStride != 0u &&
            mglCheckedNSUIntegerProduct(xfbStride,
                                        (NSUInteger)verticesPerPrimitive,
                                        &primitiveBytes) &&
            primitiveBytes > 0u;

        if (primitiveLayoutOK && xfbMTL && destinationOffsetOK &&
            requiredBytes <= remainingVisibleBytes) {
            if (!mglTessPlanBufferOrBind(
                    &executionPlan,
                    executionTemporaries, computeEncoder, xfbMTL,
                    destinationOffset, kMGLBufferSlot_IndirectParams)) {
                [self clearStageBindingCopyBacks:&stageCopyBacks];
                return false;
            }
            xfbSlot->buf->ever_written = GL_TRUE;
            xfbPrimitiveCapacity = captureVertices / verticesPerPrimitive;
            xfbWrittenBytes = xfbPrimitiveCapacity * primitiveBytes;
        } else {
            xfbTemporary = mglTessCreateBuffer(
                _device, requiredBytes, MTLResourceStorageModeShared);
            if (!xfbTemporary) {
                NSLog(@"MGL TESS XFB: failed to allocate %lu-byte overflow buffer",
                      (unsigned long)requiredBytes);
                [self clearStageBindingCopyBacks:&stageCopyBacks];
                return false;
            }
            memset(xfbTemporary.contents, 0, requiredBytes);
            if (!mglTessPlanBufferOrBind(
                    &executionPlan,
                    executionTemporaries, computeEncoder, xfbTemporary, 0,
                    kMGLBufferSlot_IndirectParams)) {
                [self clearStageBindingCopyBacks:&stageCopyBacks];
                return false;
            }

            /* Unknown layouts stay in the temporary buffer. This is an honest
             * no-capture fallback; copying an unproven stride could overwrite
             * bytes outside a complete transform-feedback primitive. */
            if (primitiveLayoutOK && xfbMTL && destinationOffsetOK &&
                remainingVisibleBytes >= primitiveBytes) {
                xfbPrimitiveCapacity = MIN(captureVertices / verticesPerPrimitive,
                                           remainingVisibleBytes / primitiveBytes);
                xfbCopyBytes = xfbPrimitiveCapacity * primitiveBytes;
                xfbWrittenBytes = xfbCopyBytes;
                xfbCopyDestination = xfbMTL;
                xfbCopyDestinationOffset = destinationOffset;
                xfbDestination = xfbSlot->buf;
            }
        }
    }

    MTLSize threadgroups = MTLSizeMake(patchCount, 1, 1);
    MTLSize threadsPerTG = MTLSizeMake(vertsPerPatch, 1, 1);
    if (!mglTessPlanDispatchOrBind(
            &executionPlan, computeEncoder,
            threadgroups, threadsPerTG)) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    /* Without this, a TES dispatch with no copy-backs stays in the current
     * command buffer and flushCommandBufferLocked's empty-CB skip drops it:
     * glFinish then never executes the TES writes (SSBO stores vanish). */
    _currentCBHasWork = YES;

    {
        MGLRenderCppCopyBackEntry copyBackEntries[kMGLMaxBufferSlots] = {0};
        uint32_t copyBackEntryCount = 0u;
        for (NSUInteger slot = 0; slot < kMGLMaxBufferSlots; slot++) {
            MGLStageBindingCopyBack *entry = &stageCopyBacks.slots[slot];
            if (entry->length == 0u) continue;
            copyBackEntries[copyBackEntryCount++] =
                (MGLRenderCppCopyBackEntry){
                    .temporary = (__bridge void *)entry->temporary,
                    .destination = (__bridge void *)entry->destination,
                    .destination_buffer = entry->destination_buffer,
                    .destination_offset = entry->destination_offset,
                    .length = entry->length,
                };
        }
        executionPlan.barrier_scope = copyBackEntryCount
            ? MGL_RENDER_CPP_COMPUTE_BARRIER_BUFFERS
            : MGL_RENDER_CPP_COMPUTE_BARRIER_NONE;
        MGLRenderCppComputeExecutionResult executionResult = {0};
        char executionError[256] = {0};
        if (mglRenderCppExecuteComputeExecutionPlan(
                _renderPassManager.state->currentCommandBufferOwner,
                _gpuRecovery.commandRecoveryOwner,
                &executionPlan, copyBackEntries, copyBackEntryCount, 0u,
                &executionResult, executionError,
                sizeof(executionError)) != 0) {
            if (executionResult.transaction.device_reset_requested) {
                atomic_store_explicit(&_deviceResetRequested, true,
                                      memory_order_release);
            }
            NSLog(@"MGL TESS ERROR: C++ TES-only execution failed: %s",
                  executionError[0] ? executionError : "unknown error");
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }
        [self clearStageBindingCopyBacks:&stageCopyBacks];
    }

    if (xfbCopyBytes > 0u) {
        const MGLRenderCppBufferCopyEntry xfbCopy = {
            .source_buffer = (__bridge void *)xfbTemporary,
            .source_offset = 0u,
            .destination_buffer = (__bridge void *)xfbCopyDestination,
            .destination_offset = xfbCopyDestinationOffset,
            .length = xfbCopyBytes,
        };
        if (!mglTessEncodeBufferCopiesForOwner(
                _renderPassManager.state->currentCommandBufferOwner,
                &xfbCopy, 1u)) {
            NSLog(@"MGL TESS XFB: failed to encode bounded copy");
            return false;
        }
        if (xfbDestination) {
            xfbDestination->ever_written = GL_TRUE;
        }
    }

    if (xfbCaptureActive && xfbWrittenBytes > 0u) {
        GLuint64 currentOffset = xfbState->buffer_write_offsets[0];
        if ((GLuint64)xfbWrittenBytes > UINT64_MAX - currentOffset) {
            xfbState->buffer_write_offsets[0] = UINT64_MAX;
        } else {
            xfbState->buffer_write_offsets[0] =
                currentOffset + (GLuint64)xfbWrittenBytes;
        }
    }

    /* Update GL_PRIMITIVES_GENERATED query by reading the tess factor buffer
     * and computing the number of primitives generated per patch.  The TES
     * compute kernel dispatch above only runs TES once per patch (not per
     * tessellated vertex), so we must manually compute the primitive count
     * that the hardware tessellator would have produced.
     *
     * MTLQuadTessellationFactorsHalf = { half edge[4]; half inside[2]; } = 12 B.
     * For triangles: primitives ≈ ceil(inside)² (rough estimate).
     * For quads:      primitives ≈ 2 × ceil(inside0) × ceil(inside1).
     * For isolines:   primitives ≈ ceil(edge[0]). */
    if (_tessellation.tessFactorBuffer) {
        const struct {
            uint16_t edge[4];
            uint16_t inside[2];
        } __attribute__((packed)) *tessFactors =
            (const void *)_tessellation.tessFactorBuffer.contents;

        GLenum genMode = tesProgram ? tesProgram->tess_gen_mode : GL_TRIANGLES;
        GLboolean pointMode = tesProgram ? tesProgram->tess_gen_point_mode : GL_FALSE;
        const GLenum spacing =
            tesProgram ? tesProgram->tess_gen_spacing : 0;

        GLuint64 totalPrimitives = 0;
        for (GLuint p = 0; p < patchCount; p++) {
            /* Tessellation factors are half-floats.  Convert to float. */
            float edge[4], inside[2];
            for (int i = 0; i < 4; i++) {
                edge[i] = *(const __fp16 *)&tessFactors[p].edge[i];
            }
            for (int i = 0; i < 2; i++) {
                inside[i] = *(const __fp16 *)&tessFactors[p].inside[i];
            }
            /* Zero/NaN factor: the patch is discarded and generates no
             * primitives (GL 4.6 §11.2.2.2). */
            if (mglRenderCppTessFactorsDiscardPatch(
                    (uint32_t)genMode, edge, inside)) {
                continue;
            }
            for (int i = 0; i < 4; i++) {
                if (edge[i] < 1.0f) edge[i] = 1.0f;
            }
            for (int i = 0; i < 2; i++) {
                if (inside[i] < 1.0f) inside[i] = 1.0f;
            }

            GLuint perPatch = 0;
            if (pointMode) {
                /* Point mode: 1 primitive per tessellated point. */
                if (genMode == GL_QUADS) {
                    perPatch =
                        mglTessRoundLevelForSpacing(
                            spacing, (GLuint)ceilf(inside[0])) *
                        mglTessRoundLevelForSpacing(
                            spacing, (GLuint)ceilf(inside[1]));
                } else if (genMode == GL_TRIANGLES) {
                    const GLuint n = mglTessRoundLevelForSpacing(
                        spacing, (GLuint)ceilf(inside[0]));
                    perPatch = n * n;
                } else { /* GL_ISOLINES */
                    perPatch = (GLuint)ceilf(edge[0]);
                }
            } else {
                if (genMode == GL_QUADS) {
                    /* Each quad splits into 2 triangles. */
                    perPatch =
                        2u * mglTessRoundLevelForSpacing(
                                 spacing, (GLuint)ceilf(inside[0])) *
                        mglTessRoundLevelForSpacing(
                            spacing, (GLuint)ceilf(inside[1]));
                } else if (genMode == GL_TRIANGLES) {
                    const GLuint n = mglTessRoundLevelForSpacing(
                        spacing, (GLuint)ceilf(inside[0]));
                    perPatch = n * n;
                } else { /* GL_ISOLINES */
                    /* Each isoline segment is 1 line primitive (2 vertices). */
                    perPatch = (GLuint)ceilf(edge[0]);
                }
            }
            if (perPatch == 0u) perPatch = 1u;
            totalPrimitives += perPatch;
        }

        GLuint64 writtenPrimitives = totalPrimitives;
        if (xfbCaptureActive && writtenPrimitives > (GLuint64)xfbPrimitiveCapacity) {
            writtenPrimitives = (GLuint64)xfbPrimitiveCapacity;
        }
        mglRecordActivePrimitiveQueryDraw(glm_ctx, totalPrimitives, writtenPrimitives);
    }

    return true;
}

@end
