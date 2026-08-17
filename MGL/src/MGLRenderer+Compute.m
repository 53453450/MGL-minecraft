// MGLRenderer+Compute.m
// Compute dispatch methods extracted from MGLRenderer.m.
// These methods do not depend on any file-scope static functions in MGLRenderer.m.

#import "MGLRenderer_Private.h"
#import "mgl_compute_pipeline_cache.h"
#include "mgl_env_flag.h"
#include "mgl_render_cpp.h"
#include "mgl_render_cpp_objc.h"

static BOOL mglComputeUsesMetalCpp(void)
{
    return mglRenderCppGetDevice() != NULL;
}

static MGLMetalBufferRef mglComputeCreateBufferWithBytes(
    MGLMetalDeviceRef device,
    const void *bytes,
    NSUInteger length,
    MTLResourceOptions options)
{
    if (mglComputeUsesMetalCpp()) {
        void *buffer = NULL;
        if (mglRenderCppCreateBufferWithBytes(bytes, length, options, NULL,
                                              &buffer) == 0 && buffer) {
            return (__bridge_transfer MGLMetalBufferRef)buffer;
        }
    }
    return [device newBufferWithBytes:bytes length:length options:options];
}

static MGLMetalSamplerStateRef mglComputeCreateSampler(
    MGLMetalDeviceRef device,
    MTLSamplerDescriptor *descriptor)
{
    if (mglComputeUsesMetalCpp()) {
        void *sampler = NULL;
        if (mglRenderCppCreateSampler((__bridge void *)descriptor,
                                      &sampler) == 0 && sampler) {
            return (__bridge_transfer MGLMetalSamplerStateRef)sampler;
        }
    }
    return [device newSamplerStateWithDescriptor:descriptor];
}

static MGLMetalTextureRef mglComputeCreateTextureLevelView(
    MGLMetalTextureRef texture,
    NSUInteger level,
    NSUInteger sliceCount)
{
    if (mglComputeUsesMetalCpp()) {
        void *view = NULL;
        if (mglRenderCppCreateTextureViewRange(
                (__bridge void *)texture, (uint32_t)texture.pixelFormat,
                (uint32_t)texture.textureType, level, 1, 0, sliceCount,
                0, 0, 0, 0, 0, &view) == 0 && view) {
            return (__bridge_transfer MGLMetalTextureRef)view;
        }
    }
    return [texture newTextureViewWithPixelFormat:texture.pixelFormat
                                      textureType:texture.textureType
                                           levels:NSMakeRange(level, 1)
                                           slices:NSMakeRange(0, sliceCount)];
}

static void mglComputeSetBuffer(MGLMetalComputeCommandEncoderRef encoder,
                                MGLMetalBufferRef buffer,
                                NSUInteger offset,
                                NSUInteger index)
{
    if (mglComputeUsesMetalCpp()) {
        (void)mglRenderCppSetComputeBuffer(
            (__bridge void *)encoder, (__bridge void *)buffer,
            (uint64_t)offset, (uint32_t)index);
        return;
    }
    [encoder setBuffer:buffer offset:offset atIndex:index];
}

static void mglComputeSetTexture(MGLMetalComputeCommandEncoderRef encoder,
                                 MGLMetalTextureRef texture,
                                 NSUInteger index)
{
    if (mglComputeUsesMetalCpp()) {
        (void)mglRenderCppSetComputeTexture(
            (__bridge void *)encoder, (__bridge void *)texture,
            (uint32_t)index);
        return;
    }
    [encoder setTexture:texture atIndex:index];
}

static void mglComputeSetSampler(MGLMetalComputeCommandEncoderRef encoder,
                                 MGLMetalSamplerStateRef sampler,
                                 NSUInteger index)
{
    if (mglComputeUsesMetalCpp()) {
        (void)mglRenderCppSetComputeSampler(
            (__bridge void *)encoder, (__bridge void *)sampler,
            (uint32_t)index);
        return;
    }
    [encoder setSamplerState:sampler atIndex:index];
}

static void mglComputeSetPipeline(MGLMetalComputeCommandEncoderRef encoder,
                                   MGLMetalComputePipelineStateRef pipeline)
{
    if (mglComputeUsesMetalCpp()) {
        (void)mglRenderCppSetComputePipelineState(
            (__bridge void *)encoder, (__bridge void *)pipeline);
        return;
    }
    [encoder setComputePipelineState:pipeline];
}

static void mglComputeDispatch(MGLMetalComputeCommandEncoderRef encoder,
                               MTLSize groups,
                               MTLSize threads)
{
    if (mglComputeUsesMetalCpp()) {
        (void)mglRenderCppDispatchCompute(
            (__bridge void *)encoder, (uint32_t)groups.width,
            (uint32_t)groups.height, (uint32_t)groups.depth,
            (uint32_t)threads.width, (uint32_t)threads.height,
            (uint32_t)threads.depth);
        return;
    }
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

static void mglComputeDispatchIndirect(MGLMetalComputeCommandEncoderRef encoder,
                                       MGLMetalBufferRef buffer,
                                       NSUInteger offset,
                                       MTLSize threads)
{
    if (mglComputeUsesMetalCpp()) {
        (void)mglRenderCppDispatchComputeIndirect(
            (__bridge void *)encoder, (__bridge void *)buffer,
            (uint64_t)offset, (uint32_t)threads.width,
            (uint32_t)threads.height, (uint32_t)threads.depth);
        return;
    }
    [encoder dispatchThreadgroupsWithIndirectBuffer:buffer
                                  indirectBufferOffset:offset
                                 threadsPerThreadgroup:threads];
}

static void mglComputeEndEncoder(MGLMetalComputeCommandEncoderRef encoder)
{
    if (mglComputeUsesMetalCpp()) {
        (void)mglRenderCppEndComputeEncoder((__bridge void *)encoder);
        return;
    }
    [encoder endEncoding];
}

@interface MGLRenderer (ComputeLocked)
- (void)mtlDispatchComputeLocked:(GLMContext)glm_ctx
                         groupsX:(GLuint)groups_x
                         groupsY:(GLuint)groups_y
                         groupsZ:(GLuint)groups_z;
- (void)mtlDispatchComputeIndirectLocked:(GLMContext)glm_ctx
                                indirect:(GLintptr)indirect;
@end

void mglRendererCallbackDispatchCompute(void *runtime_context,
                                        GLMContext glm_ctx,
                                        unsigned int groups_x,
                                        unsigned int groups_y,
                                        unsigned int groups_z)
{
    MGLRenderer *renderer = (__bridge MGLRenderer *)runtime_context;
    if (!renderer || !glm_ctx) return;
    METAL_LOCK();
    [renderer mtlDispatchComputeLocked:glm_ctx
                               groupsX:groups_x
                               groupsY:groups_y
                               groupsZ:groups_z];
    METAL_UNLOCK();
}

void mglRendererCallbackDispatchComputeIndirect(void *runtime_context,
                                                GLMContext glm_ctx,
                                                intptr_t indirect)
{
    MGLRenderer *renderer = (__bridge MGLRenderer *)runtime_context;
    if (!renderer || !glm_ctx) return;
    METAL_LOCK();
    [renderer mtlDispatchComputeIndirectLocked:glm_ctx indirect:indirect];
    METAL_UNLOCK();
}

@implementation MGLRenderer (Compute)

#pragma mark ----- compute utility ---------------------------------------------------------------------

- (bool) bindBuffersToComputeEncoder:(id <MTLComputeCommandEncoder>) computeCommandEncoder
                                stage:(int)stage
                              copyBacks:(MGLStageBindingCopyBackList *)copyBacks
{
    return [self bindBuffersToComputeEncoder:computeCommandEncoder
                                       stage:stage
                                     copyBacks:copyBacks
                                 executionPlan:NULL
                                  temporaries:nil];
}

- (bool) bindBuffersToComputeEncoder:(id <MTLComputeCommandEncoder>) computeCommandEncoder
                                stage:(int)stage
                              copyBacks:(MGLStageBindingCopyBackList *)copyBacks
                          executionPlan:(MGLRenderCppComputeExecutionPlan *)executionPlan
                           temporaries:(NSMutableArray *)temporaries
{
    if ((!computeCommandEncoder && !executionPlan) || !copyBacks) {
        NSLog(@"MGL COMPUTE ERROR: NULL compute encoder for buffer binding");
        return false;
    }

    /* P4.5（round 35）：compute 绑定 setter 序列 snapshot 化 —— 与 render
     * binding snapshot 同构的 op 列表，gate-on 收集后一次 C ABI 重放（位置在
     * 本函数末尾，保持「map 循环 emit → runtime-size buffer emit」顺序）；
     * 任一校验失败路径先 flush 已收集 op 再 return false，与直接路径「已发生
     * emit」对齐。gate-off 直接 mglComputeSetBuffer（A/B 对照）。copy-back
     * 登记等书keeping 保持内联、两门一致。 */
    const BOOL useComputeBindingSnapshot = mglComputeUsesMetalCpp() || executionPlan;
    BOOL snapshotOK = YES;
    MGLRenderCppComputeBindingSnapshot cbindSnapshot = {0};
#define MGL_CBIND_FLUSH_SNAPSHOT()                                              \
    do {                                                                        \
        if (useComputeBindingSnapshot && cbindSnapshot.op_count > 0) {          \
            if (executionPlan) {                                                \
                if (mglRenderCppAppendComputeBindingSnapshotToPlan(             \
                        executionPlan, &cbindSnapshot, NULL, 0) != 0) {         \
                    snapshotOK = NO;                                            \
                }                                                               \
            } else {                                                            \
                snapshotOK = mglRenderCppEncodeComputeBindingSnapshot(          \
                    (__bridge void *)computeCommandEncoder, &cbindSnapshot,    \
                    NULL, 0) == 0 && snapshotOK;                                 \
            }                                                                   \
            cbindSnapshot = (MGLRenderCppComputeBindingSnapshot){0};            \
        }                                                                       \
    } while (0)

#define MGL_CBIND_RETAIN_TEMP(obj)                                               \
    do {                                                                         \
        if (executionPlan && temporaries && (obj)) {                            \
            [temporaries addObject:(obj)];                                      \
        }                                                                        \
    } while (0)

#define MGL_CBIND_EMIT_BUFFER(slot, bufPtr, off)                                \
    do {                                                                        \
        if (useComputeBindingSnapshot) {                                        \
            if (cbindSnapshot.op_count >=                                       \
                MGL_RENDER_CPP_COMPUTE_BINDING_SNAPSHOT_MAX_OPS) {              \
                MGL_CBIND_FLUSH_SNAPSHOT();                                     \
            }                                                                   \
            cbindSnapshot.ops[cbindSnapshot.op_count++] =                       \
                (MGLRenderCppComputeBindingOp){/* kind */ 0u,                   \
                                               /* index */ (uint32_t)(slot),    \
                                               /* offset */ (uint64_t)(off),    \
                                               /* buffer */ (void *)(bufPtr),   \
                                               /* bytes */ NULL,                \
                                               /* length */ 0u};                \
        } else {                                                                \
            mglComputeSetBuffer(computeCommandEncoder,                          \
                                (__bridge MGLMetalBufferRef)(bufPtr), (off),        \
                                (slot));                                        \
        }                                                                       \
    } while (0)

    BufferMapList localBufferMap = {0};
    BufferMapList *bufferMap = stage == _COMPUTE_SHADER
        ? &MGL_STATE(ctx)->compute_buffer_map_list : &localBufferMap;
    RETURN_FALSE_ON_FAILURE(
        [self mapGLBuffersToMTLBufferMap:bufferMap stage:stage]);

    // dirty buffer covers all buffer modifications
    if (MGL_STATE(ctx)->dirty_bits & DIRTY_BUFFER)
    {
        // updateDirtyBaseBufferList binds new mtl buffers or updates old ones
        [self updateDirtyBaseBufferList:bufferMap];

        MGL_STATE(ctx)->dirty_bits &= ~DIRTY_BUFFER;
    }

    for(int i=0; i<bufferMap->count; i++)
    {
        BufferMap *map = &bufferMap->buffers[i];
        Buffer *ptr;
        NSUInteger metalBindingIndex;
        NSUInteger bindOffset;

        ptr = map->buf;

        if (!ptr) {
            MGL_CBIND_FLUSH_SNAPSHOT();
            NSLog(@"MGL COMPUTE ERROR: buffer map[%d] NULL buffer", i);
            return false;
        }

        metalBindingIndex = map->has_metal_binding
            ? (NSUInteger)map->metal_binding_index
            : (NSUInteger)map->buffer_base_index;
        if (metalBindingIndex >= kMGLMaxMetalVertexBufferCount) {
            NSLog(@"MGL COMPUTE WARNING: buffer map[%d] Metal slot %lu out of range, skipping",
                  i,
                  (unsigned long)metalBindingIndex);
            continue;
        }
        [self clearStageBindingCopyBack:copyBacks atIndex:metalBindingIndex];
        if (map->offset < 0) {
            NSLog(@"MGL COMPUTE WARNING: buffer map[%d] negative offset=%lld, skipping",
                  i,
                  (long long)map->offset);
            MGL_CBIND_FLUSH_SNAPSHOT();
            return false;
        }
        bindOffset = (NSUInteger)map->offset;

        /* Compute has no inline set*Bytes path, so it needs a real Metal
         * buffer.  Small plain-uniform slots deliberately do not carry one
         * (see updateDirtyBuffer); create it from the current CPU shadow
         * instead of falling through to a zero-filled isolated binding. */
        if (!ptr->data.mtl_data) {
            [self bindMTLBuffer:ptr];
        }
        MGLMetalBufferRef buffer = ptr->data.mtl_data
            ? (__bridge MGLMetalBufferRef)(ptr->data.mtl_data)
            : nil;

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
        if (needsIsolatedBinding) {
            NSUInteger fallbackLength = MAX(requiredBytes, sizeof(uint32_t));
            MGLMetalBufferRef isolated =
                [self isolatedStageBindingBufferForMap:map
                                                 source:buffer
                                         requiredLength:fallbackLength];
            if (!isolated) {
                NSLog(@"MGL COMPUTE ERROR: failed to isolate undersized buffer map[%d] buffer=%u required=%lu available=%lu",
                      i,
                      (unsigned)ptr->name,
                      (unsigned long)fallbackLength,
                      (unsigned long)availableBytes);
                MGL_CBIND_FLUSH_SNAPSHOT();
                return false;
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

            /* Isolate the undefined suffix from page-alignment bytes. A
             * post-dispatch blit preserves writes to the legal prefix. */
            MGL_CBIND_EMIT_BUFFER(metalBindingIndex,
                                 (__bridge void *)isolated, 0);
            MGL_CBIND_RETAIN_TEMP(isolated);
            /* Isolated buffers are owned only by this loop local (created via
             * __bridge_transfer on gate-on): flush immediately so the encoder
             * retains the buffer while it is still alive, instead of holding a
             * dangling pointer in the snapshot until the end-of-function
             * replay. */
            MGL_CBIND_FLUSH_SNAPSHOT();
            continue;
        }

        MGL_CBIND_EMIT_BUFFER(metalBindingIndex,
                             (__bridge void *)buffer, bindOffset);
        mglNoteBufferEncoded(ptr);
    }

    /* Bind spvBufferSizeConstants for runtime-sized SSBO arrays.
     * The AIR backend emits code that reads uint32 byte-sizes from a
     * constant uint* buffer at MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX when a
     * shader uses .length() on unsized SSBO arrays.  The pure fill (slot
     * cap / self-slot exclusion / uint32 truncation) lives in the C++
     * facade mglRenderCppBuildRuntimeArraySizes (P4.5, item 1138); the
     * ObjC side only extracts the per-buffer {slot, visible-size} pairs
     * from the GL buffer map. */
    {
        Program *computeProgram = mglResolveProgramForStageFromState(ctx, stage);
        if (computeProgram && computeProgram->modules[stage].needs_runtime_array_size_buffer)
        {
            const GLuint runtimeSizeSlot =
                mglRuntimeArraySizeBufferIndexForProgram(computeProgram, stage);
            uint32_t sizeConstants[kMGLMaxMetalVertexBufferCount];
            memset(sizeConstants, 0, sizeof(sizeConstants));

            MGLRenderCppBufferSizeEntry entries[32]; /* MAX_MAPPED_BUFFERS */
            uint32_t entryCount = 0;
            for (int i = 0; i < bufferMap->count && entryCount < 32; i++)
            {
                BufferMap *map = &bufferMap->buffers[i];
                if (!map->buf)
                    continue;
                NSUInteger metalSlot = map->has_metal_binding
                    ? (NSUInteger)map->metal_binding_index
                    : (NSUInteger)map->buffer_base_index;
                GLsizeiptr visibleSize = mglBufferMapVisibleSize(map);
                entries[entryCount].metal_slot = (uint32_t)metalSlot;
                entries[entryCount].visible_size = (uint64_t)visibleSize;
                entryCount++;
            }

            if (mglRenderCppBuildRuntimeArraySizes(
                    entries, entryCount,
                    runtimeSizeSlot,
                    kMGLMaxMetalVertexBufferCount,
                    sizeConstants, kMGLMaxMetalVertexBufferCount) != 0) {
                NSLog(@"MGL COMPUTE ERROR: runtime-array-size constants build failed");
                MGL_CBIND_FLUSH_SNAPSHOT();
                return false;
            }

            MGLMetalBufferRef sizeBuffer = mglComputeCreateBufferWithBytes(
                _device, sizeConstants, sizeof(sizeConstants),
                MTLResourceStorageModeShared);
            if (sizeBuffer) {
                MGL_CBIND_EMIT_BUFFER(runtimeSizeSlot,
                                      (__bridge void *)sizeBuffer, 0);
                MGL_CBIND_RETAIN_TEMP(sizeBuffer);
                /* sizeBuffer is a block-local (__bridge_transfer on gate-on):
                 * flush before the block ends so the encoder retains it. */
                MGL_CBIND_FLUSH_SNAPSHOT();
            }
        }
    }

    /* Flush any collected compute binding ops — the replay position (after
     * the map loop and the runtime-size buffer emit) matches the direct
     * path's encoder order exactly. */
    MGL_CBIND_FLUSH_SNAPSHOT();
#undef MGL_CBIND_EMIT_BUFFER
#undef MGL_CBIND_FLUSH_SNAPSHOT
#undef MGL_CBIND_RETAIN_TEMP
    if (!snapshotOK) {
        return false;
    }
    return true;
}

- (bool) bindTexturesToComputeEncoder:(id <MTLComputeCommandEncoder>) computeCommandEncoder
                                 stage:(int)stage
{
    return [self bindTexturesToComputeEncoder:computeCommandEncoder
                                         stage:stage
                                 executionPlan:NULL
                                  temporaries:nil];
}

- (bool) bindTexturesToComputeEncoder:(id <MTLComputeCommandEncoder>) computeCommandEncoder
                                 stage:(int)stage
                         executionPlan:(MGLRenderCppComputeExecutionPlan *)executionPlan
                          temporaries:(NSMutableArray *)temporaries
{
    GLuint count;
    enum {
        _TEXTURE,
        _IMAGE_TEXTURE
    };
    struct {
        int spvc_type;
        int gl_texture_type;
    } mapped_types[] = {
        {_SAMPLED_IMAGE_RES, _TEXTURE},
        {_STORAGE_IMAGE_RES, _IMAGE_TEXTURE},
        {0,0}
    };

    if (!computeCommandEncoder && !executionPlan) {
        NSLog(@"MGL COMPUTE ERROR: NULL compute encoder for texture binding");
        return false;
    }

    /* P4.5（round 36）：compute 纹理/sampler 绑定并入同一 binding snapshot
     * （kind 2 = texture / 3 = sampler），gate-on 收集后一次重放；gate-off
     * 直接 mglComputeSetTexture/mglComputeSetSampler（A/B 对照）。临时对象
     * （level view / fallback sampler，gate-on __bridge_transfer 局部）经
     * ctexTemporaries 强持有至末尾重放后才释放——禁止悬垂进延迟重放。 */
    const BOOL useComputeTextureSnapshot = mglComputeUsesMetalCpp() || executionPlan;
    BOOL textureSnapshotOK = YES;
    MGLRenderCppComputeBindingSnapshot ctexSnapshot = {0};
    NSMutableArray *ctexTemporaries = temporaries;
#define MGL_CTEX_RETAIN_TEMP(obj)                                               \
    do {                                                                        \
        if (useComputeTextureSnapshot && (obj)) {                               \
            if (!ctexTemporaries) ctexTemporaries = [NSMutableArray array];     \
            [ctexTemporaries addObject:(obj)];                                  \
        }                                                                       \
    } while (0)

#define MGL_CTEX_FLUSH_SNAPSHOT()                                               \
    do {                                                                        \
        if (useComputeTextureSnapshot && ctexSnapshot.op_count > 0) {           \
            if (executionPlan) {                                                \
                if (mglRenderCppAppendComputeBindingSnapshotToPlan(             \
                        executionPlan, &ctexSnapshot, NULL, 0) != 0) {          \
                    textureSnapshotOK = NO;                                     \
                }                                                               \
            } else {                                                            \
                textureSnapshotOK = mglRenderCppEncodeComputeBindingSnapshot(    \
                    (__bridge void *)computeCommandEncoder, &ctexSnapshot,     \
                    NULL, 0) == 0 && textureSnapshotOK;                          \
            }                                                                   \
            ctexSnapshot = (MGLRenderCppComputeBindingSnapshot){0};             \
        }                                                                       \
    } while (0)

#define MGL_CTEX_EMIT_TEXTURE(slot, texPtr)                                     \
    do {                                                                        \
        if (useComputeTextureSnapshot) {                                        \
            if (ctexSnapshot.op_count >=                                        \
                MGL_RENDER_CPP_COMPUTE_BINDING_SNAPSHOT_MAX_OPS) {              \
                MGL_CTEX_FLUSH_SNAPSHOT();                                      \
            }                                                                   \
            ctexSnapshot.ops[ctexSnapshot.op_count++] =                         \
                (MGLRenderCppComputeBindingOp){/* kind */ 2u,                   \
                                               /* index */ (uint32_t)(slot),    \
                                               /* offset */ 0,                  \
                                               /* buffer */ (void *)(texPtr),   \
                                               /* bytes */ NULL,                \
                                               /* length */ 0u};                \
        } else {                                                                \
            mglComputeSetTexture(computeCommandEncoder,                         \
                                 (__bridge MGLMetalTextureRef)(texPtr), (slot));    \
        }                                                                       \
    } while (0)

#define MGL_CTEX_EMIT_SAMPLER(slot, smpPtr)                                     \
    do {                                                                        \
        if (useComputeTextureSnapshot) {                                        \
            if (ctexSnapshot.op_count >=                                        \
                MGL_RENDER_CPP_COMPUTE_BINDING_SNAPSHOT_MAX_OPS) {              \
                MGL_CTEX_FLUSH_SNAPSHOT();                                      \
            }                                                                   \
            ctexSnapshot.ops[ctexSnapshot.op_count++] =                         \
                (MGLRenderCppComputeBindingOp){/* kind */ 3u,                   \
                                               /* index */ (uint32_t)(slot),    \
                                               /* offset */ 0,                  \
                                               /* buffer */ (void *)(smpPtr),   \
                                               /* bytes */ NULL,                \
                                               /* length */ 0u};                \
        } else {                                                                \
            mglComputeSetSampler(computeCommandEncoder,                         \
                                 (__bridge MGLMetalSamplerStateRef)(smpPtr),        \
                                 (slot));                                       \
        }                                                                       \
    } while (0)

    Program *computeProgram = mglResolveProgramForStageFromState(ctx, stage);

    for(int type=0; mapped_types[type].spvc_type; type++)
    {
        int spvc_type;
        int gl_texture_type;

        spvc_type = mapped_types[type].spvc_type;
        gl_texture_type = mapped_types[type].gl_texture_type;

        // iterate shader storage buffers
        count = mglRendererGetProgramBindingCount(ctx, stage, spvc_type);
        if (count)
        {
            int textures_to_be_mapped = count;

            if (textures_to_be_mapped > TEXTURE_UNITS) {
                textures_to_be_mapped = TEXTURE_UNITS;
            }

            for (int i=0; i < (int)count && textures_to_be_mapped > 0; i++)
            {
                MGLShaderResource *resource = NULL;
                GLuint metalBinding = mglRendererGetProgramBinding(ctx, stage, spvc_type, i);
                GLuint glUnit = 0u;
                Texture *ptr = NULL;

                if (computeProgram &&
                    spvc_type >= 0 && spvc_type < MGL_MAX_SHADER_RESOURCES &&
                    i >= 0 &&
                    i < (int)computeProgram->shader_resources_list[stage][spvc_type].count) {
                    resource = &computeProgram->shader_resources_list[stage][spvc_type].list[i];
                    metalBinding = mglMetalResourceSlot(resource);
                }

                if (metalBinding >= TEXTURE_UNITS ||
                    mglShouldSkipStageTextureResource(computeProgram,
                                                      stage,
                                                      spvc_type,
                                                      resource)) {
                    continue;
                }

                switch(gl_texture_type)
                {
                    case _TEXTURE:
                        glUnit = [self textureUnitForSampledResource:resource
                                                         metalBinding:metalBinding
                                                                stage:stage];
                        if (glUnit >= TEXTURE_UNITS) {
                            continue;
                        }
                        ptr = [self textureForSampledResource:resource
                                                 metalBinding:metalBinding
                                                         stage:stage
                                                  expectedType:(MTLTextureType)
                                                      mglRendererGetProgramDeclaredTextureType(ctx, stage, spvc_type, i)];
                        break;
                    case _IMAGE_TEXTURE:
                        glUnit = resource ? (resource->sampler_unit >= 0 ? (GLuint)resource->sampler_unit : resource->gl_binding)
                                          : mglRendererGetProgramGLBinding(ctx, stage, spvc_type, i);
                        if (glUnit >= TEXTURE_UNITS) {
                            continue;
                        }
                        ptr = MGL_STATE(ctx)->image_units[glUnit].tex;
                        break;
                    default:
                        ptr = NULL;
                        NSLog(@"MGL COMPUTE ERROR: unknown compute texture binding class %d", gl_texture_type);
                        MGL_CTEX_FLUSH_SNAPSHOT();
                        return false;
                }

                if (ptr)
                {
                    RETURN_FALSE_ON_FAILURE([self bindMTLTexture: ptr]);
                    if (!ptr->mtl_data) {
                        continue;
                    }

                    MGLMetalTextureRef texture;
                    texture = (__bridge MGLMetalTextureRef)(ptr->mtl_data);
                    if (!texture) {
                        continue;
                    }

                    /* For storage images bound to a non-zero mipmap level, create
                     * a level-specific texture view so imageSize() returns the
                     * dimensions at the bound level (matches the fragment-stage
                     * path).  Sampled textures are not affected. */
                    if (gl_texture_type == _IMAGE_TEXTURE) {
                        GLuint imgLevel = MGL_STATE(ctx)->image_units[glUnit].level;
                        if (imgLevel > 0u) {
                            NSUInteger sliceCount = texture.arrayLength;
                            if (texture.textureType == MTLTextureTypeCube ||
                                texture.textureType == MTLTextureTypeCubeArray) {
                                sliceCount = texture.arrayLength * 6u;
                            }
                            MGLMetalTextureRef levelView =
                                mglComputeCreateTextureLevelView(
                                    texture, imgLevel, sliceCount);
                            if (levelView) {
                                texture = levelView;
                                /* Keep the view alive until the end replay. */
                                MGL_CTEX_RETAIN_TEMP(levelView);
                            }
                        }
                    }

                    MGLMetalSamplerStateRef sampler;

                    // late binding of texture samplers.. but its better than scanning the entire texture_samplers
                    if(gl_texture_type == _TEXTURE && MGL_STATE(ctx)->texture_samplers[glUnit])
                    {
                        Sampler *gl_sampler;

                        gl_sampler = MGL_STATE(ctx)->texture_samplers[glUnit];

                        // delete existing sampler if dirty
                        if (gl_sampler->dirty_bits)
                        {
                            if (gl_sampler->mtl_data)
                            {
                                mglSafeReleaseMetalObj((void **)&gl_sampler->mtl_data);
                            }
                        }

                        if (gl_sampler->mtl_data == NULL)
                        {
                            gl_sampler->mtl_data = (void *)CFBridgingRetain([self createMTLSamplerForTexParam:&gl_sampler->params target:ptr->target]);
                            gl_sampler->dirty_bits = 0;
                        }

                        sampler = (__bridge MGLMetalSamplerStateRef)(gl_sampler->mtl_data);
                    }
                    else
                    {
                        sampler = (__bridge MGLMetalSamplerStateRef)(ptr->params.mtl_data);
                    }

                    if (!sampler) {
                        MGLMetalSamplerStateRef fallbackSampler =
                            mglComputeCreateSampler(_device,
                                                    [MTLSamplerDescriptor new]);
                        sampler = fallbackSampler;
                        /* Keep the fallback alive until the end replay. */
                        MGL_CTEX_RETAIN_TEMP(sampler);
                        if (!sampler) {
                            continue;
                        }
                    }

                    MGL_CTEX_EMIT_TEXTURE(metalBinding,
                                          (__bridge void *)texture);
                    if (gl_texture_type == _TEXTURE &&
                        (!resource || resource->has_combined_sampler)) {
                        GLuint samplerBinding = resource
                            ? mglMetalCombinedSamplerSlot(resource)
                            : metalBinding;
                        MGL_CTEX_EMIT_SAMPLER(samplerBinding,
                                              (__bridge void *)sampler);
                    }

                    textures_to_be_mapped--;
                }
            }

            // texture not found
            if (textures_to_be_mapped)
            {
                DEBUG_PRINT("No texture bound for fragment shader location\n");
                MGL_CTEX_FLUSH_SNAPSHOT();
                return false;
            }
        }
    }

    if (computeProgram) {
        MGLShaderResourceList *arrayResources =
            &computeProgram->shader_resources_list[stage][_SAMPLED_IMAGE_RES];
        for (GLuint resourceIndex = 0; arrayResources->list && resourceIndex < arrayResources->count; resourceIndex++) {
            MGLShaderResource *resource = &arrayResources->list[resourceIndex];
            if (resource->gl_array_size <= 1) {
                continue;
            }

            MTLTextureType expectedType = (MTLTextureType)
                mglRendererGetProgramDeclaredTextureType(ctx, stage, _SAMPLED_IMAGE_RES, (int)resourceIndex);
            for (GLint element = 1; element < resource->gl_array_size; element++) {
                GLuint metalSlot = resource->binding + (GLuint)element;
                GLuint samplerSlot =
                    mglMetalCombinedSamplerSlotForElement(resource,
                                                          (GLuint)element);
                if (metalSlot >= TEXTURE_UNITS) {
                    break;
                }

                GLuint glUnit = [self textureUnitForSampledResource:NULL
                                                        metalBinding:metalSlot
                                                               stage:stage];
                Texture *ptr = [self textureForSampledResource:NULL
                                                   metalBinding:metalSlot
                                                           stage:stage
                                                    expectedType:expectedType];
                if (!ptr || ![self bindMTLTexture:ptr] || !ptr->mtl_data) {
                    continue;
                }

                MGLMetalTextureRef texture = (__bridge MGLMetalTextureRef)(ptr->mtl_data);
                MGLMetalSamplerStateRef sampler = nil;
                if (glUnit < TEXTURE_UNITS && MGL_STATE(ctx)->texture_samplers[glUnit]) {
                    Sampler *glSampler = MGL_STATE(ctx)->texture_samplers[glUnit];
                    if (glSampler->mtl_data == NULL) {
                        glSampler->mtl_data = (void *)CFBridgingRetain(
                            [self createMTLSamplerForTexParam:&glSampler->params target:ptr->target]);
                        glSampler->dirty_bits = 0;
                    }
                    sampler = (__bridge MGLMetalSamplerStateRef)(glSampler->mtl_data);
                } else if (ptr->params.mtl_data) {
                    sampler = (__bridge MGLMetalSamplerStateRef)(ptr->params.mtl_data);
                }
                if (!sampler) {
                    sampler = mglComputeCreateSampler(
                        _device, [MTLSamplerDescriptor new]);
                    /* Keep the fallback alive until the end replay. */
                    MGL_CTEX_RETAIN_TEMP(sampler);
                }

                MGL_CTEX_EMIT_TEXTURE(metalSlot,
                                      (__bridge void *)texture);
                if (resource->has_combined_sampler && sampler) {
                    MGL_CTEX_EMIT_SAMPLER(samplerSlot,
                                          (__bridge void *)sampler);
                }
            }
        }
    }

    /* Bind additional array elements for storage image arrays.
     * The AIR backend emits `array<texture2d<T, access::read_write>, N> image [[texture(B)]]`
     * which occupies consecutive Metal texture slots B..B+N-1.  The main
     * loop above only binds element 0; bind elements 1..N-1 here. */
    if (computeProgram) {
        MGLShaderResourceList *storageArrayResources =
            &computeProgram->shader_resources_list[stage][_STORAGE_IMAGE_RES];
        for (GLuint resourceIndex = 0; storageArrayResources->list && resourceIndex < storageArrayResources->count; resourceIndex++) {
            MGLShaderResource *resource = &storageArrayResources->list[resourceIndex];
            if (resource->gl_array_size <= 1) {
                continue;
            }

            for (GLint element = 1; element < resource->gl_array_size; element++) {
                GLuint metalSlot = resource->binding + (GLuint)element;
                if (metalSlot >= TEXTURE_UNITS) {
                    break;
                }

                GLuint glUnit = (resource->sampler_unit >= 0 ? (GLuint)resource->sampler_unit : resource->gl_binding) + (GLuint)element;
                if (glUnit >= TEXTURE_UNITS) {
                    continue;
                }

                Texture *ptr = MGL_STATE(ctx)->image_units[glUnit].tex;
                if (!ptr || ![self bindMTLTexture:ptr] || !ptr->mtl_data) {
                    continue;
                }

                MGLMetalTextureRef texture = (__bridge MGLMetalTextureRef)(ptr->mtl_data);

                /* For storage images bound to a non-zero mipmap level, create
                 * a level-specific texture view (matches element 0 path). */
                GLuint imgLevel = MGL_STATE(ctx)->image_units[glUnit].level;
                if (imgLevel > 0u) {
                    NSUInteger sliceCount = texture.arrayLength;
                    if (texture.textureType == MTLTextureTypeCube ||
                        texture.textureType == MTLTextureTypeCubeArray) {
                        sliceCount = texture.arrayLength * 6u;
                    }
                    MGLMetalTextureRef levelView =
                        mglComputeCreateTextureLevelView(
                            texture, imgLevel, sliceCount);
                    if (levelView) {
                        texture = levelView;
                        /* Keep the view alive until the end replay. */
                        MGL_CTEX_RETAIN_TEMP(levelView);
                    }
                }

                MGL_CTEX_EMIT_TEXTURE(metalSlot,
                                      (__bridge void *)texture);
            }
        }
    }

    /* Flush any collected texture/sampler ops — the replay position (after
     * the array passes) matches the direct path's encoder order. */
    MGL_CTEX_FLUSH_SNAPSHOT();
#undef MGL_CTEX_EMIT_TEXTURE
#undef MGL_CTEX_EMIT_SAMPLER
#undef MGL_CTEX_FLUSH_SNAPSHOT
#undef MGL_CTEX_RETAIN_TEMP
    ctexTemporaries = nil;

    MGL_STATE(ctx)->dirty_bits &= ~(DIRTY_TEX_BINDING | DIRTY_SAMPLER | DIRTY_IMAGE_UNIT_STATE);

    if (!textureSnapshotOK) {
        return false;
    }

    return true;
}

#pragma mark ------------------------------------------------------------------------------------------
#pragma mark processCompute
#pragma mark ------------------------------------------------------------------------------------------
- (bool)processCompute:(id <MTLComputeCommandEncoder>)computeCommandEncoder
             copyBacks:(MGLStageBindingCopyBackList *)copyBacks
{
    return [self processCompute:computeCommandEncoder
                       copyBacks:copyBacks
                   executionPlan:NULL
                    temporaries:nil];
}

- (bool)processCompute:(id <MTLComputeCommandEncoder>)computeCommandEncoder
             copyBacks:(MGLStageBindingCopyBackList *)copyBacks
         executionPlan:(MGLRenderCppComputeExecutionPlan *)executionPlan
          temporaries:(NSMutableArray *)temporaries
{
    // from https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Compute-Ctx/Compute-Ctx.html#//apple_ref/doc/uid/TP40014221-CH6-SW1
    Program *program;

    if (!computeCommandEncoder && !executionPlan) {
        NSLog(@"MGL COMPUTE ERROR: processCompute called with NULL encoder");
        return false;
    }

    program = mglResolveProgramForStageFromState(ctx, _COMPUTE_SHADER);
    if (!program) {
        NSLog(@"MGL COMPUTE ERROR: glDispatchCompute with no current program");
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return false;
    }

    if (program->dirty_bits)
    {
        if (![self bindMTLProgram: program]) {
            NSLog(@"MGL COMPUTE ERROR: failed to bind compute program %u", program->name);
            return false;
        }
    }

    Shader *computeShader;
    computeShader = program->shader_slots[_COMPUTE_SHADER];
    if (!computeShader) {
        NSLog(@"MGL COMPUTE ERROR: current program %u has no compute shader", program->name);
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return false;
    }

    id <MTLFunction> func;
    func = (__bridge MGLMetalFunctionRef)(program->modules[_COMPUTE_SHADER].mtl_function);
    if (!func) {
        NSLog(@"MGL COMPUTE ERROR: compute shader for program %u has no Metal function", program->name);
        return false;
    }

    void *computePipelineHandle = NULL;
    char computePipelineError[512] = {0};
    int computePipelineResult = mglGetOrCreateProgramComputePipeline(
        program, _COMPUTE_SHADER, &computePipelineHandle,
        computePipelineError, sizeof(computePipelineError));
    MGLMetalComputePipelineStateRef computePipelineState =
        computePipelineResult == 0 && computePipelineHandle
            ? (__bridge_transfer MGLMetalComputePipelineStateRef)computePipelineHandle
            : nil;
    if (!computePipelineState) {
        NSLog(@"MGL COMPUTE ERROR: failed to create compute pipeline for program %u: %s",
              program->name,
              computePipelineError[0] ? computePipelineError : "unknown error");
        return false;
    }

    if (executionPlan) {
        executionPlan->pipeline = (__bridge void *)computePipelineState;
        if (temporaries) {
            [temporaries addObject:computePipelineState];
        }
    } else {
        mglComputeSetPipeline(computeCommandEncoder, computePipelineState);
    }

    RETURN_FALSE_ON_FAILURE([self bindBuffersToComputeEncoder:computeCommandEncoder
                                                         stage:_COMPUTE_SHADER
                                                      copyBacks:copyBacks
                                                  executionPlan:executionPlan
                                                   temporaries:temporaries]);

    //setTexture:atIndex:
    //setTextures:withRange:
    RETURN_FALSE_ON_FAILURE(
        [self bindTexturesToComputeEncoder:computeCommandEncoder
                                      stage:_COMPUTE_SHADER
                              executionPlan:executionPlan
                               temporaries:temporaries]);

    // setSamplerState:atIndex:
    // setSamplerState:lodMinClamp:lodMaxClamp:atIndex:
    // setSamplerStates:withRange:
    // setSamplerStates:lodMinClamps:lodMaxClamps:withRange:

    // [computeCommandEncoder setThreadgroupMemoryLength:atIndex:

    MGL_STATE(ctx)->dirty_bits = 0;

    return true;
}

-(void)mtlDispatchCompute:(GLMContext)glm_ctx groupsX:(GLuint)groups_x groupsY:(GLuint)groups_y groupsZ:(GLuint)groups_z
{
    METAL_LOCK();
    [self mtlDispatchComputeLocked:glm_ctx
                           groupsX:groups_x
                           groupsY:groups_y
                           groupsZ:groups_z];
    METAL_UNLOCK();
}

/* P4.5 (item 1147): 两条 compute dispatch 入口（direct/indirect）共享的
 * 编排主体——endRenderEncoding → ensureWritableCommandBuffer → 纹理绑定 →
 * compute encoder 创建 → processCompute → 程序解析 → 按 dispatch 类型编码
 * → encoder 结束 → copy-back flush → dirty bits。错误路径按调用点语义清理
 * 中间态（encoder 结束、copy-backs 清空）。返回 YES 成功 / NO 失败。 */
- (BOOL)runComputeDispatchOrchestrationLocked:(GLMContext)glm_ctx
                                  dispatchKind:(uint32_t)dispatchKind
                                     groupsX:(GLuint)groups_x
                                     groupsY:(GLuint)groups_y
                                     groupsZ:(GLuint)groups_z
                              indirectBuffer:(MGLMetalBufferRef)indirectBuffer
                              indirectOffset:(NSUInteger)indirectOffset
                                      reason:(const char *)reason
{
    // end encoding on current render encoder
    [self endRenderEncoding];

    if (![self ensureWritableCommandBuffer:reason]) {
        return NO;
    }

    for (NSUInteger unit = 0; unit < TEXTURE_UNITS; unit++) {
        Texture *imageTexture = MGL_STATE(glm_ctx)->image_units[unit].tex;
        if (imageTexture) {
            if (![self bindMTLTexture:imageTexture]) {
                return NO;
            }
        }

        Texture *sampledTexture = MGL_STATE(glm_ctx)->active_textures[unit];
        if (sampledTexture) {
            if (![self bindMTLTexture:sampledTexture]) {
                return NO;
            }
        }
    }

    MGLStageBindingCopyBackList copyBacks = {0};
    const BOOL useExecutionPlan = mglComputeUsesMetalCpp();
    MGLRenderCppComputeExecutionPlan executionPlan = {0};
    NSMutableArray *executionTemporaries = useExecutionPlan
        ? [NSMutableArray array] : nil;
    MGLMetalComputeCommandEncoderRef computeCommandEncoder = nil;
    if (!useExecutionPlan) {
        computeCommandEncoder =
            mglRenderCreateComputeEncoderForCommandBufferOwner(
                _renderPassManager.state->currentCommandBufferOwner);
        if (!computeCommandEncoder) {
            NSLog(@"MGL ERROR: Failed to create compute command encoder for %s",
                  reason ? reason : "dispatch");
            return NO;
        }
    }

    if (![self processCompute:computeCommandEncoder
                    copyBacks:&copyBacks
                executionPlan:useExecutionPlan ? &executionPlan : NULL
                 temporaries:executionTemporaries]) {
        if (computeCommandEncoder) {
            mglComputeEndEncoder(computeCommandEncoder);
        }
        [self clearStageBindingCopyBacks:&copyBacks];
        return NO;
    }

    Program *ptr;
    ptr = mglResolveProgramForStageFromState(glm_ctx, _COMPUTE_SHADER);
    if (!ptr) {
        NSLog(@"MGL COMPUTE ERROR: %s with no current compute program after binding",
              reason ? reason : "glDispatchCompute");
        if (computeCommandEncoder) {
            mglComputeEndEncoder(computeCommandEncoder);
        }
        [self clearStageBindingCopyBacks:&copyBacks];
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    /* P4.5: dispatch 参数 value-state plan —— ObjC 只传 groups + 未解析的
     * local size（0 由 C++ 解析为 1，与 `x ? x : 1` 默认一致），gate-on
     * 一次 C ABI 调用在 C++ 内完成 dispatchThreadgroups 编码；gate-off 走
     * 原逐条 ObjC 路径作 A/B 对照。 */
    BOOL hasCopyBackEntries = NO;
    if (useExecutionPlan) {
        /* Copy-back resources are consumed by a following blit/CPU-visible
         * transaction. Request an explicit buffer barrier at the end of the
         * compute encoder so the plan carries the visibility requirement
         * alongside dispatch and binding state. */
        for (NSUInteger slot = 0; slot < kMGLMaxBufferSlots; slot++) {
            if (copyBacks.slots[slot].length != 0) {
                hasCopyBackEntries = YES;
                break;
            }
        }
        executionPlan.barrier_scope = hasCopyBackEntries
            ? MGL_RENDER_CPP_COMPUTE_BARRIER_BUFFERS
            : MGL_RENDER_CPP_COMPUTE_BARRIER_NONE;
        executionPlan.dispatch = (MGLRenderCppComputePlan){
            .dispatch_kind = dispatchKind,
            .groups_x = groups_x,
            .groups_y = groups_y,
            .groups_z = groups_z,
            .local_x = ptr->local_workgroup_size.x,
            .local_y = ptr->local_workgroup_size.y,
            .local_z = ptr->local_workgroup_size.z,
            .indirect_buffer = indirectBuffer
                ? (__bridge void *)indirectBuffer : NULL,
            .indirect_offset = indirectOffset,
        };
        MGLRenderCppCopyBackEntry copyBackEntries[kMGLMaxBufferSlots] = {0};
        uint32_t copyBackEntryCount = 0;
        for (NSUInteger slot = 0; slot < kMGLMaxBufferSlots; slot++) {
            MGLStageBindingCopyBack *entry = &copyBacks.slots[slot];
            if (entry->length == 0) continue;
            copyBackEntries[copyBackEntryCount++] =
                (MGLRenderCppCopyBackEntry){
                    .temporary = (__bridge void *)entry->temporary,
                    .destination = (__bridge void *)entry->destination,
                    .destination_buffer = entry->destination_buffer,
                    .destination_offset = entry->destination_offset,
                    .length = entry->length,
                };
        }
        MGLRenderCppComputeExecutionResult executionResult = {0};
        char executionError[256] = {0};
        if (mglRenderCppExecuteComputeExecutionPlan(
                _renderPassManager.state->currentCommandBufferOwner,
                _gpuRecovery.commandRecoveryOwner,
                &executionPlan,
                copyBackEntries,
                copyBackEntryCount,
                0u,
                &executionResult,
                executionError,
                sizeof(executionError)) != 0) {
            if (executionResult.transaction.device_reset_requested) {
                atomic_store_explicit(&_deviceResetRequested, true,
                                      memory_order_release);
            }
            NSLog(@"MGL COMPUTE ERROR: C++ %s execution transaction failed: %s",
                  reason ? reason : "dispatch",
                  executionError[0] ? executionError : "unknown error");
            [self clearStageBindingCopyBacks:&copyBacks];
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return NO;
        }
        [self clearStageBindingCopyBacks:&copyBacks];
    } else {
        MTLSize numThreadgroups;
        MTLSize threadsPerThreadgroup;
        if (dispatchKind == MGL_RENDER_CPP_COMPUTE_DISPATCH_DIRECT) {
            numThreadgroups = MTLSizeMake(groups_x, groups_y, groups_z);
        } else {
            numThreadgroups = MTLSizeMake(0, 0, 0);
        }
        /* P4.5 (item 1141/887): 线程组尺寸 0->1 默认在 C++
         * （mglRenderCppThreadgroupSize，两门共用）。 */
        MGLRenderCppThreadgroupSize tg = {0};
        mglRenderCppThreadgroupSize(
            ptr->local_workgroup_size.x, ptr->local_workgroup_size.y,
            ptr->local_workgroup_size.z, &tg);
        threadsPerThreadgroup = MTLSizeMake(tg.x, tg.y, tg.z);
        if (dispatchKind == MGL_RENDER_CPP_COMPUTE_DISPATCH_DIRECT) {
            mglComputeDispatch(computeCommandEncoder, numThreadgroups,
                               threadsPerThreadgroup);
        } else {
            mglComputeDispatchIndirect(computeCommandEncoder, indirectBuffer,
                                       indirectOffset, threadsPerThreadgroup);
        }
    }

    if (computeCommandEncoder) {
        mglComputeEndEncoder(computeCommandEncoder);
    }
    /* Without this, a dispatch with no copy-backs stays in the current
     * command buffer and flushCommandBufferLocked's empty-CB skip drops it:
     * glFinish then never executes the compute writes (SSBO stores vanish). */
    _currentCBHasWork = YES;

    if (!useExecutionPlan &&
        ![self flushStageBindingCopyBacks:&copyBacks
                     requireCPUVisibility:NO]) {
        NSLog(@"MGL COMPUTE ERROR: failed to copy isolated writable buffer prefixes after %s",
              reason ? reason : "dispatch");
        mglDispatchError(glm_ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return NO;
    }
    if (useExecutionPlan && hasCopyBackEntries &&
        ![self newCommandBufferLocked]) {
        NSLog(@"MGL COMPUTE ERROR: failed to install post-compute command buffer after %s",
              reason ? reason : "dispatch");
        return NO;
    }

    /* Fine-grained dirty bits instead of DIRTY_ALL.  Compute dispatch
     * ends the render encoder, so the next draw must rebuild it.  DIRTY_STATE
     * triggers newRenderEncoderLocked; DIRTY_FBO re-syncs the render pass;
     * the remaining bits (matching kMGLFullReplayDirtyBits in MGLRenderer+Draw.m)
     * re-bind all GL resources that the render encoder needs.  DIRTY_SHADER and
     * DIRTY_DRAWABLE are intentionally excluded — DIRTY_SHADER is a per-program
     * bit, and DIRTY_DRAWABLE only applies at context init. */
    mglMarkRendererDirtyBits(
        glm_ctx->active_state,
        DIRTY_STATE | DIRTY_FBO | DIRTY_PROGRAM | DIRTY_VAO |
        DIRTY_RENDER_STATE | DIRTY_TEX_BINDING | DIRTY_TEX |
        DIRTY_TEX_PARAM | DIRTY_SAMPLER | DIRTY_ALPHA_STATE |
        DIRTY_BUFFER | DIRTY_BUFFER_BASE_STATE | DIRTY_IMAGE_UNIT_STATE);
    return YES;
}

-(void)mtlDispatchComputeLocked:(GLMContext)glm_ctx groupsX:(GLuint)groups_x groupsY:(GLuint)groups_y groupsZ:(GLuint)groups_z
{
    if (!glm_ctx) {
        NSLog(@"MGL COMPUTE ERROR: mtlDispatchCompute called with NULL context");
        return;
    }

    ctx = glm_ctx;

    if (groups_x == 0 || groups_y == 0 || groups_z == 0) {
        NSLog(@"MGL COMPUTE TRACE: glDispatchCompute zero-sized dispatch %ux%ux%u skipped",
              groups_x,
              groups_y,
              groups_z);
        return;
    }

    /* P4.5 (item 1147): 共享编排主体在
     * runComputeDispatchOrchestrationLocked:。 */
    if (![self runComputeDispatchOrchestrationLocked:glm_ctx
                                        dispatchKind:MGL_RENDER_CPP_COMPUTE_DISPATCH_DIRECT
                                           groupsX:groups_x
                                           groupsY:groups_y
                                           groupsZ:groups_z
                                    indirectBuffer:nil
                                    indirectOffset:0
                                            reason:"glDispatchCompute"]) {
        return;
    }

    for (NSUInteger unit = 0; unit < TEXTURE_UNITS; unit++) {
        ImageUnit *imageUnit = &MGL_STATE(glm_ctx)->image_units[unit];
        Texture *imageTexture = imageUnit->tex;
        if (!imageTexture ||
            (imageUnit->access != GL_WRITE_ONLY && imageUnit->access != GL_READ_WRITE)) {
            continue;
        }
        imageTexture->metal_data_authoritative = GL_TRUE;
        if (imageTexture->faces[0].levels &&
            imageUnit->level >= 0 &&
            imageUnit->level < (GLint)imageTexture->num_levels) {
            imageTexture->faces[0].levels[imageUnit->level].metal_data_authoritative = GL_TRUE;
        }
    }

    //[self newRenderEncoder];
}


-(void)mtlDispatchComputeIndirect:(GLMContext)glm_ctx indirect:(GLintptr)indirect
{
    METAL_LOCK();
    [self mtlDispatchComputeIndirectLocked:glm_ctx indirect:indirect];
    METAL_UNLOCK();
}

-(void)mtlDispatchComputeIndirectLocked:(GLMContext)glm_ctx indirect:(GLintptr)indirect
{
    if (!glm_ctx) {
        NSLog(@"MGL COMPUTE ERROR: mtlDispatchComputeIndirect called with NULL context");
        return;
    }

    ctx = glm_ctx;

    Buffer *glIndirectBuffer = MGL_STATE(glm_ctx)->buffers[_DISPATCH_INDIRECT_BUFFER];
    if (MGL_STATE(glm_ctx)->var.dispatch_indirect_buffer_binding == 0 || !glIndirectBuffer) {
        NSLog(@"MGL COMPUTE ERROR: glDispatchComputeIndirect with no GL_DISPATCH_INDIRECT_BUFFER bound");
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }
    if (indirect < 0) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_VALUE);
        return;
    }

    if (![self processBuffer:glIndirectBuffer]) {
        NSLog(@"MGL COMPUTE ERROR: failed to process dispatch indirect buffer %u",
              glIndirectBuffer ? glIndirectBuffer->name : 0u);
        return;
    }

    MGLMetalBufferRef indirectBuffer = (__bridge MGLMetalBufferRef)(glIndirectBuffer->data.mtl_data);
    if (!indirectBuffer) {
        NSLog(@"MGL COMPUTE ERROR: dispatch indirect buffer %u has no Metal backing",
              glIndirectBuffer ? glIndirectBuffer->name : 0u);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    NSUInteger indirectOffset = (NSUInteger)indirect;
    NSUInteger indirectArgBytes = 3u * sizeof(uint32_t);
    if (indirectOffset > indirectBuffer.length ||
        indirectArgBytes > (indirectBuffer.length - indirectOffset)) {
        NSLog(@"MGL COMPUTE ERROR: dispatch indirect range exceeds Metal buffer buffer=%u off=%lu bytes=%lu len=%lu",
              glIndirectBuffer ? glIndirectBuffer->name : 0u,
              (unsigned long)indirectOffset,
              (unsigned long)indirectArgBytes,
              (unsigned long)indirectBuffer.length);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    /* P4.5 (item 1147): 共享编排主体在
     * runComputeDispatchOrchestrationLocked:。 */
    if (![self runComputeDispatchOrchestrationLocked:glm_ctx
                                        dispatchKind:MGL_RENDER_CPP_COMPUTE_DISPATCH_INDIRECT
                                           groupsX:0
                                           groupsY:0
                                           groupsZ:0
                                    indirectBuffer:indirectBuffer
                                    indirectOffset:indirectOffset
                                            reason:"glDispatchComputeIndirect"]) {
        return;
    }
}

@end
