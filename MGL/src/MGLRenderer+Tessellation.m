/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

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
#include "mgl_shader_abi.h"
#include "mgl_air_gs_abi.h"
#include "mgl_air_tess_abi.h"
#include "mgl_draw_tess.h"

extern void mglRecordActivePrimitiveQueryDraw(GLMContext ctx, GLuint64 generated, GLuint64 written);

enum {
    MGL_TESS_RESOURCE_STORAGE_SHARED = 0u,
    MGL_TESS_COMMAND_STATUS_NOT_ENQUEUED = 0u,
    MGL_TESS_COMMAND_STATUS_COMMITTED = 2u,
    MGL_TESS_TEXTURE_TYPE_CUBE = 5u,
    MGL_TESS_TEXTURE_TYPE_CUBE_ARRAY = 6u,
};

static id mglTessCreateBuffer(id device,
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

static id mglTessCreateBufferWithBytes(
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

static id mglTessCreateSampler(id device)
{
    (void)device;
    void *sampler = NULL;
    if (mglRenderCreateDefaultSampler(&sampler) == 0 && sampler) {
        return (__bridge_transfer id)sampler;
    }
    return nil;
}

static uint64_t mglTessBufferLength(id buffer)
{
    MGLRenderBufferInfo info = {0};
    return buffer && mglRenderGetBufferInfo((__bridge void *)buffer, &info) == 0
        ? info.length : 0u;
}

static void *mglTessBufferContents(id buffer)
{
    void *contents = NULL;
    uint64_t length = 0u;
    return buffer &&
        mglRenderGetBufferContents((__bridge void *)buffer,
                                      &contents, &length) == 0
        ? contents : NULL;
}

static bool mglTessTextureInfo(id texture, MGLRenderTextureInfo *info)
{
    return texture && info &&
        mglRenderGetTextureInfo((__bridge void *)texture, info) == 0;
}

static id mglTessCreateTextureLevelView(
    id texture,
    NSUInteger level,
    NSUInteger sliceCount)
{
    MGLRenderTextureInfo info = {0};
    if (!mglTessTextureInfo(texture, &info)) return nil;
    void *view = NULL;
    if (mglRenderCreateTextureViewRange(
            (__bridge void *)texture, info.pixel_format,
            info.texture_type, level, 1, 0, sliceCount,
            0, 0, 0, 0, 0, &view) == 0 && view) {
        return (__bridge_transfer id)view;
    }
    return nil;
}

static void mglTessSetRenderVertexBuffer(id encoder,
                                         void *renderEncoderOwner,
                                         id buffer,
                                         NSUInteger offset,
                                         NSUInteger index)
{
    (void)encoder;
    (void)mglRenderSetRenderBufferForOwner(
        renderEncoderOwner, (__bridge void *)buffer, offset,
        MGL_RENDER_BINDING_STAGE_VERTEX, (uint32_t)index);
}

static void mglTessDrawPrimitives(id encoder,
                                  void *renderEncoderOwner,
                                  uint32_t type,
                                  NSUInteger vertexStart,
                                  NSUInteger vertexCount,
                                  NSUInteger instanceCount,
                                  NSUInteger baseInstance)
{
    const MGLRenderDrawPlan plan = {
            .kind = MGL_RENDER_DRAW_ARRAY,
            .primitive_type = (uint32_t)type,
            .vertex_start = vertexStart,
            .vertex_count = vertexCount,
            .instance_count = instanceCount,
            .base_instance = baseInstance,
        };
    (void)encoder;
    (void)mglRenderEncodeDrawForRenderEncoderOwner(
        renderEncoderOwner, &plan, NULL, 0);
}

static bool mglTessEncodeBufferCopiesForOwner(
    void *commandBufferOwner,
    const MGLRenderBufferCopyEntry *entries,
    uint32_t entryCount)
{
    if (!commandBufferOwner || !entries || entryCount == 0u) return false;
    return mglRenderEncodeBufferCopiesForCommandBufferOwner(
        commandBufferOwner, entries, entryCount) == 0;
}

static bool mglTessAppendComputeResourceOp(
    MGLRenderComputeExecutionPlan *plan,
    NSMutableArray *temporaries,
    uint32_t kind,
    id resource,
    NSUInteger offset,
    NSUInteger index)
{
    if (!plan || kind > 3u) {
        return false;
    }
    if (plan->binding_op_count >= MGL_RENDER_COMPUTE_EXECUTION_MAX_OPS) {
        NSLog(@"MGL TESS ERROR: compute binding op overflow (%u)",
              (unsigned)plan->binding_op_count);
        return false;
    }
    plan->binding_ops[plan->binding_op_count++] =
        (MGLRenderComputeBindingOp){
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
    MGLRenderComputeExecutionPlan *plan,
    NSMutableArray *temporaries,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    if (!plan || !temporaries || !bytes || length == 0u ||
        length > UINT32_MAX) {
        return false;
    }
    if (plan->binding_op_count >= MGL_RENDER_COMPUTE_EXECUTION_MAX_OPS) {
        NSLog(@"MGL TESS ERROR: compute bytes-binding overflow (%u)",
              (unsigned)plan->binding_op_count);
        return false;
    }
    NSData *storage = [NSData dataWithBytes:bytes length:length];
    if (!storage) return false;
    [temporaries addObject:storage];
    plan->binding_ops[plan->binding_op_count++] =
        (MGLRenderComputeBindingOp){
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
    MGLRenderComputeExecutionPlan *plan,
    NSMutableArray *temporaries,
    id encoder,
    id buffer,
    NSUInteger offset,
    NSUInteger index)
{
    (void)encoder;
    return mglTessAppendComputeResourceOp(
        plan, temporaries, 0u, buffer, offset, index);
}

static bool mglTessPlanTextureOrBind(
    MGLRenderComputeExecutionPlan *plan,
    NSMutableArray *temporaries,
    id encoder,
    id texture,
    NSUInteger index)
{
    (void)encoder;
    return mglTessAppendComputeResourceOp(
        plan, temporaries, 2u, texture, 0u, index);
}

static bool mglTessPlanSamplerOrBind(
    MGLRenderComputeExecutionPlan *plan,
    NSMutableArray *temporaries,
    id encoder,
    id sampler,
    NSUInteger index)
{
    (void)encoder;
    return mglTessAppendComputeResourceOp(
        plan, temporaries, 3u, sampler, 0u, index);
}

static bool mglTessPlanBytesOrBind(
    MGLRenderComputeExecutionPlan *plan,
    NSMutableArray *temporaries,
    id encoder,
    const void *bytes,
    NSUInteger length,
    NSUInteger index)
{
    (void)encoder;
    return mglTessAppendComputeBytesOp(
        plan, temporaries, bytes, length, index);
}

static bool mglTessPlanDispatchOrBind(
    MGLRenderComputeExecutionPlan *plan,
    id encoder,
    uint32_t groupsX,
    uint32_t groupsY,
    uint32_t groupsZ,
    uint32_t localX,
    uint32_t localY,
    uint32_t localZ)
{
    MGLRenderComputePlan dispatch = {
        .dispatch_kind = MGL_RENDER_COMPUTE_DISPATCH_DIRECT,
        .groups_x = groupsX,
        .groups_y = groupsY,
        .groups_z = groupsZ,
        .local_x = localX,
        .local_y = localY,
        .local_z = localZ,
        .indirect_buffer = NULL,
        .indirect_offset = 0u,
    };
    (void)encoder;
    if (plan &&
        plan->dispatch_op_count >= MGL_RENDER_COMPUTE_EXECUTION_MAX_DISPATCHES) {
        NSLog(@"MGL TESS ERROR: compute dispatch sequence overflow (%u)",
              (unsigned)plan->dispatch_op_count);
        return false;
    }
    return mglRenderAppendComputeDispatchToPlan(
        plan, &dispatch, NULL, 0) == 0;
}

static const uint8_t *mglRendererReadableBufferBytes(Buffer *buffer)
{
    if (!buffer) {
        return NULL;
    }
    if (buffer->data.buffer_data &&
        mglRenderCPUPointerUsable(buffer->data.buffer_data)) {
        return (const uint8_t *)(uintptr_t)buffer->data.buffer_data;
    }
    if (buffer->data.mtl_data) {
        id mtlBuffer = (__bridge id)(buffer->data.mtl_data);
        return (const uint8_t *)mglTessBufferContents(mtlBuffer);
    }
    return NULL;
}

@implementation MGLRenderer (Tessellation)

typedef struct {
    id __strong buffer;
    NSUInteger offset;
    id __strong initialization_source;
    NSUInteger initialization_source_offset;
    NSUInteger initialization_length;
    BOOL valid;
} MGLTessStageBufferBinding;

typedef struct {
    MGLTessStageBufferBinding slots[kMGLMaxBufferSlots];
    id __strong size_buffer;
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

        uint32_t metalBindingIndex = 0u;
        if (!mglRenderResolveMappedBufferSlot(
                map->has_metal_binding ? 1 : 0, (int32_t)map->metal_binding_index,
                (int32_t)map->buffer_base_index,
                (uint32_t)kMGLMaxMetalVertexBufferCount, &metalBindingIndex)) {
            continue;
        }
        [self clearStageBindingCopyBack:copyBacks atIndex:metalBindingIndex];
        id buffer = ptr->data.mtl_data
            ? (__bridge id)(ptr->data.mtl_data)
            : nil;
        if (buffer && mglRenderBufferHasCPUDirty(ptr->data.dirty_bits)) {
            /* Consume the CPU-side initialization before a tessellation
             * stage can write the same Metal backing. Otherwise a later
             * stage bind would upload the stale shadow over the GPU result. */
            if (![self updateDirtyBuffer:ptr]) {
                return false;
            }
            buffer = ptr->data.mtl_data
                ? (__bridge id)(ptr->data.mtl_data)
                : nil;
        }
        MGLTessIsolatedBindingPlan bindPlan = {0};
        GLsizeiptr storageRemaining = mglBufferMapStorageRemaining(map);
        const uint64_t bufferLength = mglTessBufferLength(buffer);
        NSUInteger availableBytes = buffer
            ? mglBufferMapVisibleBackingBytes(map, bufferLength)
            : 0u;
        NSUInteger requiredBytes =
            mglRendererGetProgramBindingRequiredSize(ctx, stage, (int)map->resource_type, (int)map->resource_index);
        requiredBytes = mglTessRequiredBindingBytes((int)map->resource_type,
                                                    (uint32_t)requiredBytes);
        if (!mglTessPlanIsolatedBinding(
                buffer != nil, map->offset, bufferLength,
                (int64_t)storageRemaining, (uint64_t)availableBytes,
                (uint32_t)requiredBytes, (int)map->resource_type,
                &bindPlan)) {
            return false;
        }

        MGLTessStageBufferBinding *binding = &bindings->slots[metalBindingIndex];
        binding->buffer = nil;
        binding->offset = 0u;
        binding->initialization_source = nil;
        binding->initialization_source_offset = 0u;
        binding->initialization_length = 0u;
        binding->valid = YES;
        if (!bindPlan.isolated) {
            binding->buffer = buffer;
            binding->offset = (NSUInteger)map->offset;
            /* The GL buffer's Metal backing is about to be staged in a
             * compute encoder: pin its snapshot-pool slot. */
            mglNoteBufferEncoded(ptr);
            continue;
        }

        NSUInteger fallbackLength = bindPlan.fallback_length;
        id isolated = mglTessCreateBuffer(
            _device, fallbackLength, MGL_TESS_RESOURCE_STORAGE_SHARED);
        void *isolatedContents = mglTessBufferContents(isolated);
        if (!isolatedContents) {
            return false;
        }
        memset(isolatedContents, 0, fallbackLength);

        binding->buffer = isolated;
        binding->offset = 0u;
        if (bindPlan.init_length > 0u) {
            binding->initialization_source = buffer;
            binding->initialization_source_offset = (NSUInteger)map->offset;
            binding->initialization_length = bindPlan.init_length;
        }

        if (mglTessIsolatedNeedsCopyBack(bindPlan.writable ? 1 : 0,
                                         buffer ? 1 : 0,
                                         bindPlan.init_length) &&
            ![self recordStageBindingCopyBack:copyBacks
                                       atIndex:metalBindingIndex
                                     temporary:isolated
                                   destination:buffer
                             destinationBuffer:ptr
                            destinationOffset:(NSUInteger)map->offset
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
        mglTessFillRuntimeArraySizeConstants(
            stageBufferMap.buffers, stageBufferMap.count,
            bindings->size_buffer_index, sizeConstants, kMGLMaxBufferSlots);
        bindings->size_buffer = mglTessCreateBufferWithBytes(
            _device, sizeConstants, sizeof(sizeConstants),
            MGL_TESS_RESOURCE_STORAGE_SHARED);
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
    MGLRenderCommandBufferState commandState = {0};
    const int hasCommandState = mglRenderCommandBufferOwnerHasState(
        _renderPassManager.state->currentCommandBufferOwner, &commandState);
    if (!mglTessCommandBufferCanInitBlit(hasCommandState, commandState.status)) {
        return false;
    }

    MGLRenderBufferCopyEntry copyEntries[kMGLMaxBufferSlots] = {0};
    uint32_t copyEntryCount = 0u;
    for (NSUInteger i = 0; i < kMGLMaxBufferSlots; i++) {
        MGLTessStageBufferBinding *binding = &bindings->slots[i];
        if (binding->initialization_length == 0) {
            continue;
        }
        copyEntries[copyEntryCount++] = (MGLRenderBufferCopyEntry){
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
                           toComputeEncoder:(id)computeCommandEncoder
                              executionPlan:(MGLRenderComputeExecutionPlan *)executionPlan
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


- (void)bindPointSizeParamsToComputeEncoder:(id)computeEncoder
                                    program:(Program *)program
                                      stage:(int)stage
                              executionPlan:(MGLRenderComputeExecutionPlan *)executionPlan
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
    float pointSizeParams[2] = {0.f, 0.f};
    mglTessFillPointSizeParams(
        ctx && MGL_STATE(ctx)->var.point_size > 0.0f
            ? MGL_STATE(ctx)->var.point_size
            : 0.0f,
        ctx && MGL_STATE(ctx)->caps.program_point_size ? 1 : 0,
        pointSizeParams);
    (void)mglTessAppendComputeBytesOp(
        executionPlan, temporaries, pointSizeParams,
        sizeof(pointSizeParams), kMGLPointSizeParamBufferIndex);
}

- (BOOL)ensureTessTextureMetalData:(const MGLTessTextureBind *)binds
                             count:(uint32_t)count
                               ctx:(GLMContext)drawCtx
{
    if (!binds || !drawCtx || !drawCtx->active_state) {
        return YES;
    }
    for (uint32_t i = 0; i < count; i++) {
        const GLuint unit = binds[i].gl_unit;
        Texture *ptr = (binds[i].kind == MGL_TESS_BIND_STORAGE_IMAGE)
            ? MGL_STATE(drawCtx)->image_units[unit].tex
            : MGL_STATE(drawCtx)->active_textures[unit];
        if (ptr && !ptr->mtl_data) {
            [self bindMTLTexture:ptr];
        }
    }
    return YES;
}

- (BOOL)planTessTextureBinds:(const MGLTessTextureBind *)binds
                       count:(uint32_t)count
                         ctx:(GLMContext)drawCtx
                        plan:(MGLRenderComputeExecutionPlan *)plan
                 temporaries:(NSMutableArray *)temporaries
{
    if (!binds || !plan || !drawCtx || !drawCtx->active_state) {
        return binds == NULL || count == 0u;
    }
    for (uint32_t i = 0; i < count; i++) {
        const MGLTessTextureBind *bind = &binds[i];
        id texture = nil;
        Texture *ptr = NULL;
        if (bind->kind == MGL_TESS_BIND_STORAGE_IMAGE) {
            ptr = MGL_STATE(drawCtx)->image_units[bind->gl_unit].tex;
            if (ptr) {
                texture = (__bridge id)(ptr->mtl_data);
                texture = (__bridge id)mglRendererStorageImageTexture(
                    (__bridge void *)texture,
                    &MGL_STATE(drawCtx)->image_units[bind->gl_unit]);
            }
        } else {
            ptr = MGL_STATE(drawCtx)->active_textures[bind->gl_unit];
            texture = ptr ? (__bridge id)(ptr->mtl_data) : nil;
        }
        if (!mglTessPlanTextureOrBind(plan, temporaries, nil, texture,
                                      bind->metal_slot)) {
            return NO;
        }
        if (bind->kind != MGL_TESS_BIND_SAMPLED_IMAGE ||
            bind->combined_sampler_slot == UINT32_MAX) {
            continue;
        }
        id sampler = nil;
        if (MGL_STATE(drawCtx)->texture_samplers[bind->gl_unit]) {
            Sampler *glSampler =
                MGL_STATE(drawCtx)->texture_samplers[bind->gl_unit];
            if (glSampler->dirty_bits && glSampler->mtl_data) {
                mglSafeReleaseMetalObj((void **)&glSampler->mtl_data);
            }
            if (!glSampler->mtl_data && ptr) {
                glSampler->mtl_data = (void *)CFBridgingRetain(
                    [self createMTLSamplerForTexParam:&glSampler->params
                                               target:ptr->target]);
                glSampler->dirty_bits = 0;
            }
            sampler = (__bridge id)(glSampler->mtl_data);
        } else if (ptr && ptr->params.mtl_data) {
            sampler = (__bridge id)(ptr->params.mtl_data);
        }
        if (!sampler) {
            sampler = mglTessCreateSampler(_device);
        }
        if (sampler &&
            !mglTessPlanSamplerOrBind(plan, temporaries, nil, sampler,
                                      bind->combined_sampler_slot)) {
            return NO;
        }
    }
    return YES;
}

- (id)newTCSStageInBufferForContext:(GLMContext)drawCtx
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

    MGLTessTCSStageInPlan stagePlan = {0};
    if (!mglTessPlanTCSStageIn(patchVertices, patchCount, count, &stagePlan)) {
        return nil;
    }
    NSUInteger tcsInStride = (NSUInteger)stagePlan.stride;
    NSUInteger memberCount = stagePlan.member_count;
    MGLTessStageInMember members[MAX_ATTRIBS];
    memset(members, 0, sizeof(members));
    members[0] = stagePlan.members[0];
    NSUInteger tcsInVertices = (NSUInteger)stagePlan.vertices;

    VertexArray *vao = mglRendererGetValidatedVAO(drawCtx, "tcs.stage_in");
    if (!vao) {
        return nil;
    }

    const uint8_t *indexBytes = NULL;
    NSUInteger indexOffset = (NSUInteger)(uintptr_t)indices;
    uint32_t restartIndex = 0u;
    bool primitiveRestart = false;
    if (indexType != 0u) {
        Buffer *ebo = getElementBuffer(drawCtx);
        if (!ebo || ![self processBuffer:ebo]) {
            NSLog(@"MGL TESS WARNING: TCS indexed stage_in has no readable element buffer");
            return nil;
        }
        const uint8_t *eboBytes = mglRendererReadableBufferBytes(ebo);
        MGLTessIndexedStageInPlan idxPlan = {0};
        if (!eboBytes ||
            !mglTessPlanIndexedStageIn((uint32_t)indexType, (uint64_t)indexOffset,
                                       (int32_t)count, ebo->size, &idxPlan) ||
            idxPlan.status != MGL_TESS_INDEXED_STAGE_IN_OK) {
            NSLog(@"MGL TESS WARNING: TCS indexed stage_in element range OOB offset=%lu size=%lld",
                  (unsigned long)indexOffset, (long long)ebo->size);
            return nil;
        }
        indexBytes = eboBytes + indexOffset;
        primitiveRestart = mglPrimitiveRestartIndexForType(drawCtx, indexType, &restartIndex);
    }

    NSUInteger tcsInSize = (NSUInteger)stagePlan.bytes;
    id stageInBuffer = mglTessCreateBuffer(
        _device, tcsInSize, MGL_TESS_RESOURCE_STORAGE_SHARED);
    void *stageInContents = mglTessBufferContents(stageInBuffer);
    if (!stageInContents) {
        return nil;
    }
    if (!mglTessInitStageInDefaults(stageInContents, tcsInVertices,
                                    tcsInStride)) {
        return nil;
    }

    if (memberCount == 0u) {
        if (outStride) {
            *outStride = tcsInStride;
        }
        return stageInBuffer;
    }

    MGLTessStageInAttribSrc srcs[MAX_ATTRIBS];
    memset(srcs, 0, sizeof(srcs));
    for (NSUInteger m = 0; m < memberCount; m++) {
        const MGLTessStageInMember *member = &members[m];
        if (member->attribute >= MAX_ATTRIBS) {
            continue;
        }
        const VertexAttrib *attrib = &vao->attrib[member->attribute];
        MGLResolvedVertexAttribBinding resolved = {0};
        bool hasBinding = mglRendererResolveVertexAttribBinding(
            drawCtx, vao, member->attribute, "tcs.stage_in", &resolved);
        bool useCurrentValue = mglTessStageInUseCurrentValue(
            vao->enabled_attribs, member->attribute, hasBinding ? 1 : 0) != 0;
        srcs[m].type = attrib->type;
        srcs[m].attrib_size = attrib->size;
        srcs[m].normalized = attrib->normalized;
        if (useCurrentValue) {
            srcs[m].use_current = 1u;
            if (mglRendererBuildCurrentVertexAttribBytes(
                    drawCtx, member->attribute, attrib, srcs[m].current) > 0u) {
                srcs[m].current_valid = 1u;
            }
        } else if (hasBinding) {
            Buffer *vbo = resolved.buffer;
            if (vbo && [self processBuffer:vbo]) {
                srcs[m].bytes = mglRendererReadableBufferBytes(vbo);
                srcs[m].stride = resolved.stride;
                srcs[m].divisor = resolved.divisor;
                srcs[m].binding_offset = resolved.binding_offset;
                srcs[m].relativeoffset = resolved.relativeoffset;
                srcs[m].buffer_size = vbo->size >= 0 ? (uint64_t)vbo->size : 0u;
            }
        }
    }
    if (!mglTessPackStageInRecords(
            stageInContents, tcsInVertices, tcsInStride, first, count,
            indexBytes, indexType, primitiveRestart, restartIndex, baseVertex,
            baseInstance, members, (uint32_t)memberCount, srcs)) {
        return nil;
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
    id tcsPipeline =
        tcsPipelineResult == 0 && tcsPipelineHandle
            ? (__bridge_transfer id)tcsPipelineHandle
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
    if (mglRenderEncoderOwnerHasCurrent(
            _renderPassManager.state->currentRenderEncoderOwner) == 1) {
        [self endRenderEncoding];
    }

    /* Ensure a writable command buffer exists.  The GL_PATCHES path returns
     * before processGLState() (which normally creates the command buffer),
     * and prior operations (glBufferData, glEndQuery, etc.) may have
     * committed the previous command buffer. */
    MGLRenderCommandBufferState commandState = {0};
    const int hasCommandState = mglRenderCommandBufferOwnerHasState(
        _renderPassManager.state->currentCommandBufferOwner, &commandState);
    if (mglTessCommandBufferNeedsNew(hasCommandState, commandState.status)) {
        if (![self newCommandBuffer]) {
            NSLog(@"MGL TESS ERROR: failed to create command buffer for TCS dispatch");
            return false;
        }
    }

    MGLTessTextureBind tcsTextureBinds[TEXTURE_UNITS * 2u];
    const uint32_t tcsTextureBindCount = mglTessCollectTextureBinds(
        glm_ctx, tcsProgram, _TESS_CONTROL_SHADER, tcsTextureBinds,
        (uint32_t)(sizeof(tcsTextureBinds) / sizeof(tcsTextureBinds[0])));
    if (![self ensureTessTextureMetalData:tcsTextureBinds
                                    count:tcsTextureBindCount
                                      ctx:glm_ctx]) {
        return false;
    }

    MGLStageBindingCopyBackList stageCopyBacks = {0};
    MGLTessStageBufferBindingList stageBufferBindings = {0};
    if (![self prepareTessStageBufferBindings:&stageBufferBindings
                                         stage:_TESS_CONTROL_SHADER
                                     copyBacks:&stageCopyBacks]) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    MGLRenderComputeExecutionPlan executionPlan = {0};
    NSMutableArray *executionTemporaries = [NSMutableArray array];
    id computeEncoder = nil;
    executionPlan.pipeline = (__bridge void *)tcsPipeline;

    if (![self planTessTextureBinds:tcsTextureBinds
                              count:tcsTextureBindCount
                                ctx:glm_ctx
                               plan:&executionPlan
                        temporaries:executionTemporaries]) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
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

    MGLTessTCSCoreLayout tcsLayout;
    if (!mglTessComputeTCSCoreLayout(tcsProgram, contract, &tcsLayout)) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    _tessellation.tcsOutputStride = tcsLayout.output_stride;
    _tessellation.tcsOutVertices = tcsLayout.tcs_out_vertices;
    const GLuint patchVertices = tcsLayout.patch_vertices;
    const GLuint instanceCount = tcsLayout.instance_count;
    const GLuint patchCount = tcsLayout.patch_count;

    id tcsOutputBuffer = mglTessCreateBuffer(
        _device, (NSUInteger)tcsLayout.output_bytes,
        MGL_TESS_RESOURCE_STORAGE_SHARED);
    (void)mglRendererBackendSetTcsOutputBuffer(
        _backend, (__bridge void *)tcsOutputBuffer);
    void *tcsOutputContents = mglTessBufferContents(tcsOutputBuffer);
    if (!tcsOutputContents) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    memset(tcsOutputContents, 0, (size_t)tcsLayout.output_bytes);
    _tessellation.tcsOutputOffset = 0u;
    [executionTemporaries addObject:tcsOutputBuffer];

    id tcsPatchOutBuffer = mglTessCreateBuffer(
        _device, (NSUInteger)tcsLayout.patch_out_bytes,
        MGL_TESS_RESOURCE_STORAGE_SHARED);
    (void)mglRendererBackendSetTcsPatchOutBuffer(
        _backend, (__bridge void *)tcsPatchOutBuffer);
    void *tcsPatchOutContents = mglTessBufferContents(tcsPatchOutBuffer);
    if (!tcsPatchOutContents) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    memset(tcsPatchOutContents, 0, (size_t)tcsLayout.patch_out_bytes);
    [executionTemporaries addObject:tcsPatchOutBuffer];

    GLuint indirectParams[2] = {0u, 0u};
    mglTessFillTCSIndirectParams(patchVertices, instanceCount, indirectParams);
    id indirectBuf = mglTessCreateBufferWithBytes(
        _device, indirectParams, sizeof(indirectParams),
        MGL_TESS_RESOURCE_STORAGE_SHARED);
    if (!indirectBuf) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    [executionTemporaries addObject:indirectBuf];

    id tessFactorBuf = mglTessCreateBuffer(
        _device, (NSUInteger)tcsLayout.factor_bytes,
        MGL_TESS_RESOURCE_STORAGE_SHARED);
    void *tessFactorContents = mglTessBufferContents(tessFactorBuf);
    if (!tessFactorContents) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    /* GL 4.6 §11.2.2: TCS-unwritten tess levels take PATCH_DEFAULT_*. */
    if (mglRenderFillDefaultTessFactorBuffer(
            tessFactorContents, tcsLayout.factor_bytes,
            MGL_STATE(glm_ctx)->var.patch_default_outer_level,
            MGL_STATE(glm_ctx)->var.patch_default_inner_level,
            tcsLayout.patch_count) != 0) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    [executionTemporaries addObject:tessFactorBuf];

    NSUInteger tcsInStride = 0u;
    id tcsStageInBuffer =
        (__bridge id)
            mglRendererBackendGetTessVertexCaptureBuffer(_backend);
    NSUInteger tcsStageInOffset = _tessellation.tessVertexCaptureOffset;
    MGLTessTCSStageInSourcePlan stageInSource = {0};
    mglTessPlanTCSStageInSource(tcsStageInBuffer ? 1 : 0, tcsProgram,
                                &stageInSource);
    if (stageInSource.kind == MGL_TESS_TCS_STAGE_IN_CAPTURE) {
        tcsInStride = (NSUInteger)stageInSource.stride;
        [executionTemporaries addObject:tcsStageInBuffer];
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
        if (tcsStageInBuffer) {
            [executionTemporaries addObject:tcsStageInBuffer];
        }
    }
    if (!tcsStageInBuffer) {
        NSLog(@"MGL TESS WARNING: failed to pack TCS stage_in buffer for program %u",
              tcsProgram ? (unsigned)tcsProgram->name : 0u);
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    if (!mglTessAppendTCSCoreBindings(
            &executionPlan, (__bridge void *)tcsOutputBuffer,
            (__bridge void *)tcsPatchOutBuffer, (__bridge void *)indirectBuf,
            (__bridge void *)tessFactorBuf, (__bridge void *)tcsStageInBuffer,
            (uint64_t)tcsStageInOffset, &tcsLayout)) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    {
        MGLRenderCopyBackEntry copyBackEntries[kMGLMaxBufferSlots] = {0};
        uint32_t copyBackEntryCount = mglRenderCollectCopyBackEntries(
            (const MGLRenderCopyBackEntry *)stageCopyBacks.slots,
            kMGLMaxBufferSlots, copyBackEntries, kMGLMaxBufferSlots);
        executionPlan.barrier_scope = MGL_RENDER_COMPUTE_BARRIER_BUFFERS;
        MGLRenderComputeExecutionResult executionResult = {0};
        char executionError[256] = {0};
        if (mglRenderExecuteComputeExecutionPlan(
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
    (void)mglRendererBackendSetCurrentTessFactorBuffer(
        _backend, (__bridge void *)tessFactorBuf);

    return true;
}


static NSUInteger mglTESXFBFieldByteSize(GLenum glType)
{
    return (NSUInteger)mglRenderTESXFBFieldByteSize((uint64_t)glType);
}


static NSUInteger mglTESXFBVertexStride(const Program *program)
{
    return (NSUInteger)mglRenderTESXFBVertexStride((const void *)program);
}


static bool mglCheckedNSUIntegerProduct(NSUInteger a,
                                        NSUInteger b,
                                        NSUInteger *result)
{
    uint64_t out = 0u;
    if (mglRenderCheckedProduct((uint64_t)a, (uint64_t)b, &out) != 0) {
        return false;
    }
    *result = (NSUInteger)out;
    return true;
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
    id tesPipeline =
        tesPipelineResult == 0 && tesPipelineHandle
            ? (__bridge_transfer id)tesPipelineHandle
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
    id tcsOutputBuffer = (__bridge id)
        mglRendererBackendGetTcsOutputBuffer(_backend);
    id tessFactorBuffer = (__bridge id)
        mglRendererBackendGetCurrentTessFactorBuffer(_backend);
    id captureBuffer = (__bridge id)
        mglRendererBackendGetTessVertexCaptureBuffer(_backend);
    MGLTessEvalGlInPlan glInPlan = {0};
    if (!mglTessResolveEvalGlIn(
            contract, tcsOutputBuffer != nil,
            (uint64_t)_tessellation.tcsOutputOffset,
            (uint64_t)_tessellation.tcsOutputStride,
            _tessellation.tcsOutVertices, captureBuffer != nil,
            (uint64_t)_tessellation.tessVertexCaptureOffset,
            _tessellation.tessIndexedDraw ? 1 : 0,
            (uint32_t)_tessellation.tessInstanceRecords,
            (uint32_t)instanceCount, &glInPlan)) {
        NSLog(@"MGL TESS ERROR: missing TES compute inputs program=%u",
              (unsigned)tesProgram->name);
        return false;
    }
    id glInBuffer = glInPlan.from_tcs ? tcsOutputBuffer : captureBuffer;
    NSUInteger glInOffset = (NSUInteger)glInPlan.gl_in_offset;
    NSUInteger glInStride = (NSUInteger)glInPlan.gl_in_stride;
    GLuint glInVertices = glInPlan.gl_in_vertices;
    if (!mglTessEvalInputsReady(glInBuffer != nil, tessFactorBuffer != nil)) {
        NSLog(@"MGL TESS ERROR: missing TES compute inputs program=%u",
              (unsigned)tesProgram->name);
        return false;
    }
    id controlPointIndexBuffer =
        (__bridge id)
            mglRendererBackendGetTessControlPointIndexBuffer(_backend);
    if (!mglTessEvalIndexedGatherReady(
            _tessellation.tessIndexedDraw ? 1 : 0,
            controlPointIndexBuffer != nil,
            (uint32_t)_tessellation.tessInstanceRecords)) {
        NSLog(@"MGL TESS ERROR: indexed TES compute missing gather "
              "program=%u", (unsigned)tesProgram->name);
        return false;
    }
    const BOOL glInFromTCS = glInPlan.from_tcs != 0u;
    /* TCS currently expands one instance of control points / factors.
     * TES still loops instances for XFB/output bases.  Reusing instance-0
     * TCS outs is wrong when VS outputs vary by gl_InstanceID.  Until
     * per-instance TCS re-dispatch exists: one-shot log, and hard-fail when
     * MGL_TESS_MULTI_INSTANCE_ERROR is set. */
    if (mglTessMultiInstanceTCSReuseWarn(glInFromTCS ? 1 : 0,
                                         (int32_t)instanceCount)) {
        static BOOL s_multiInstanceTCSLogged = NO;
        if (!s_multiInstanceTCSLogged) {
            NSLog(@"MGL TESS ERROR: multi-instance TES with TCS reuses "
                  "instance-0 control points (program=%u instances=%d); "
                  "set MGL_TESS_MULTI_INSTANCE_ERROR=1 to fail the draw",
                  (unsigned)tesProgram->name, (int)instanceCount);
            s_multiInstanceTCSLogged = YES;
        }
        if (mglTessMultiInstanceTCSReuseIsError(glInFromTCS ? 1 : 0,
                                                (int32_t)instanceCount)) {
            return false;
        }
    }
    const NSUInteger glInInstanceStride =
        (NSUInteger)glInPlan.gl_in_instance_stride;

    /* Compute per-patch item counts and the per-instance total. */
    const uint16_t *factorBytes =
        (const uint16_t *)mglTessBufferContents(tessFactorBuffer);
    MGLTessEvalComputePlan evalPlan = {0};
    if (!mglTessPlanEvalCompute(tesProgram, factorBytes,
                                mglTessBufferLength(tessFactorBuffer),
                                patchCount, (uint32_t)instanceCount,
                                &evalPlan)) {
        NSLog(@"MGL TESS ERROR: TES compute plan failed program=%u",
              (unsigned)tesProgram->name);
        return false;
    }
    if (evalPlan.empty) {
        /* Every patch discarded (outer ≤ 0, e.g. CTS isolines with
         * outer=-1).  Empty expansion is success — do not raise
         * GL_INVALID_OPERATION. */
        return true;
    }
    const GLuint instanceCountU = evalPlan.instance_count;
    const GLuint itemsPerInstanceU = evalPlan.items_per_instance;
    NSUInteger outStride = evalPlan.out_stride;
    NSUInteger outSize = (NSUInteger)evalPlan.out_size;
    id outBuffer = mglTessCreateBuffer(
        _device, outSize, MGL_TESS_RESOURCE_STORAGE_SHARED);
    void *outContents = mglTessBufferContents(outBuffer);
    if (!outContents) {
        NSLog(@"MGL TESS ERROR: failed to allocate TES compute output "
              "(%lu bytes) program=%u",
              (unsigned long)outSize, (unsigned)tesProgram->name);
        return false;
    }
    if (mglTessSeedEvalOutputRecords(tesProgram, factorBytes, patchCount,
                                     instanceCountU, outContents, outSize,
                                     (uint32_t)outStride) != itemsPerInstanceU) {
        NSLog(@"MGL TESS ERROR: TES domain seed failed program=%u",
              (unsigned)tesProgram->name);
        return false;
    }

    /* PASS 1: pre-resolve textures before opening the compute encoder. */
    if (mglRenderEncoderOwnerHasCurrent(
            _renderPassManager.state->currentRenderEncoderOwner) == 1) {
        [self endRenderEncoding];
    }
    MGLRenderCommandBufferState commandState = {0};
    const int hasCommandState = mglRenderCommandBufferOwnerHasState(
        _renderPassManager.state->currentCommandBufferOwner, &commandState);
    if (mglTessCommandBufferNeedsNew(hasCommandState, commandState.status)) {
        if (![self newCommandBuffer]) {
            NSLog(@"MGL TESS ERROR: failed to create command buffer for TES compute");
            return false;
        }
    }

    MGLTessTextureBind tesTextureBinds[TEXTURE_UNITS * 2u];
    const uint32_t tesTextureBindCount = mglTessCollectTextureBinds(
        glm_ctx, tesProgram, _TESS_EVALUATION_SHADER, tesTextureBinds,
        (uint32_t)(sizeof(tesTextureBinds) / sizeof(tesTextureBinds[0])));
    if (![self ensureTessTextureMetalData:tesTextureBinds
                                    count:tesTextureBindCount
                                      ctx:glm_ctx]) {
        return false;
    }

    MGLStageBindingCopyBackList stageCopyBacks = {0};
    MGLTessStageBufferBindingList stageBufferBindings = {0};
    if (![self prepareTessStageBufferBindings:&stageBufferBindings
                                         stage:_TESS_EVALUATION_SHADER
                                     copyBacks:&stageCopyBacks]) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }

    MGLRenderComputeExecutionPlan executionPlan = {0};
    NSMutableArray *executionTemporaries = [NSMutableArray array];
    id computeEncoder = nil;
    executionPlan.pipeline = (__bridge void *)tesPipeline;
    id patchInputs = (__bridge id)
        mglRendererBackendGetTcsPatchOutBuffer(_backend);
    if (!mglTessPlanBufferOrBind(
            &executionPlan,
            executionTemporaries, computeEncoder,
            tessFactorBuffer, 0u,
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

    if (![self planTessTextureBinds:tesTextureBinds
                              count:tesTextureBindCount
                                ctx:glm_ctx
                               plan:&executionPlan
                        temporaries:executionTemporaries]) {
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
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


    /* Transform-feedback stream (slot 31): the kernel writes complete stage
     * records. The renderer gathers selected varyings into the compact GL XFB
     * layout and copies only the prefix containing complete primitives. */
    TransformFeedback *xfbState = MGL_STATE(glm_ctx)->transform_feedback;
    Program *gsProgram =
        mglResolveProgramForStageFromState(glm_ctx, _GEOMETRY_SHADER);
    const bool xfbActive = mglTessEvalOwnsXFB(glm_ctx, gsProgram);
    id xfbTemporary = nil;
    id xfbCopyDestination = nil;
    Buffer *xfbDestination = NULL;
    NSUInteger xfbCopyDestinationOffset = 0u;
    NSUInteger xfbCompactStride = 0u;
    NSUInteger xfbCopiedVertices = 0u;
    NSUInteger xfbWrittenBytes = 0u;
    int xfbSizeOK = 0;
    if (xfbActive) {
        BufferBaseTarget *xfbSlot =
            &MGL_STATE(glm_ctx)->buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[0];
        NSUInteger captureVertices = 0u;
        NSUInteger requiredBytes = 0u;
        const bool sessionOffsetOK =
            xfbState->buffer_write_offsets[0] <= (GLuint64)NSUIntegerMax;
        const NSUInteger xfbSessionOffset =
            sessionOffsetOK ? (NSUInteger)xfbState->buffer_write_offsets[0] : 0u;
        xfbCompactStride = mglTESXFBVertexStride(tesProgram);
        uint32_t captureVertsU = 0u;
        uint32_t requiredBytesU = 0u;
        const bool sizeOK = mglTessPlanEvalXfbCapture(
                                itemsPerInstanceU, instanceCountU,
                                (uint32_t)outStride, (uint32_t)xfbCompactStride,
                                &captureVertsU, &requiredBytesU) != 0;
        xfbSizeOK = sizeOK ? 1 : 0;
        captureVertices = captureVertsU;
        requiredBytes = requiredBytesU;
        (void)captureVertices;

        id xfbMTL = nil;
        NSUInteger visibleBytes = 0u;
        if (xfbSlot->buf) {
            if (mglRenderBufferNeedsCPUUpload(
                    xfbSlot->buf->size, xfbSlot->buf->data.dirty_bits)) {
                /* Consume CPU initialization before the XFB blit writes the
                 * same backing. Otherwise a later map can upload the stale
                 * shadow over the captured GPU data. */
                if (![self updateDirtyBuffer:xfbSlot->buf]) {
                    [self clearStageBindingCopyBacks:&stageCopyBacks];
                    return false;
                }
            } else if (xfbSlot->buf->size == 0) {
                mglRenderClearEmptyBufferDirty(xfbSlot->buf);
            }
            if (!xfbSlot->buf->data.mtl_data) {
                [self bindMTLBuffer:xfbSlot->buf];
            }
            xfbMTL = (__bridge id)(xfbSlot->buf->data.mtl_data);
            if (xfbMTL) {
                BufferMap xfbMap = {0};
                xfbMap.buf = xfbSlot->buf;
                xfbMap.offset = xfbSlot->offset;
                xfbMap.size = xfbSlot->size;
                visibleBytes =
                    mglBufferMapVisibleBackingBytes(
                        &xfbMap, (size_t)mglTessBufferLength(xfbMTL));
            }
        }

        if (mglTessPlanEvalXFBSlot(xfbActive ? 1 : 0, xfbSizeOK) ==
            MGL_TESS_EVAL_XFB_CAPTURE) {
            const GLuint verticesPerPrimitive =
                mglTessVerticesPerPrimitive(tesProgram);
            MGLTessXFBDestPlan destPlan = {0};
            const int destPlanOK =
                mglTessPlanXFBDestination(
                    itemsPerInstanceU, instanceCountU,
                    (uint32_t)xfbCompactStride, verticesPerPrimitive,
                    (uint64_t)xfbSessionOffset, (int64_t)xfbSlot->offset,
                    (uint64_t)visibleBytes, &destPlan) &&
                destPlan.valid;
            const int destOK = mglTessEvalXFBDestReady(
                xfbMTL != nil, xfbSlot->buf != NULL, destPlanOK);
            /* The AIR kernel writes full stage records (built-ins followed by
             * location-based user outputs). GL XFB is a compact stream of only
             * the selected varyings, so it can never target the GL range
             * directly. Gather the selected fields after the dispatch. */
            xfbTemporary = mglTessCreateBuffer(
                _device, requiredBytes, MGL_TESS_RESOURCE_STORAGE_SHARED);
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
            if (destOK) {
                xfbCopiedVertices = destPlan.copied_vertices;
                xfbWrittenBytes = destPlan.written_bytes;
                xfbCopyDestination = xfbMTL;
                xfbCopyDestinationOffset = destPlan.destination_offset;
                xfbDestination = xfbSlot->buf;
            }
        }
    }
    if (mglTessPlanEvalXFBSlot(xfbActive ? 1 : 0, xfbSizeOK) ==
        MGL_TESS_EVAL_XFB_DUMMY) {
        /* The TES compute kernel always declares and writes the XFB stream
         * slot (31); bind a 1-byte dummy so the slot is never dangling when
         * GL feedback is inactive. */
        const uint64_t dummyBytes = mglTessDummyXfbBytes((uint64_t)outSize);
        void *cachedDummy = NULL;
        id xfbDummy = nil;
        if (mglRendererBackendGetTessXfbDummyBuffer(
                _backend, dummyBytes, &cachedDummy) == 1) {
            xfbDummy = (__bridge id)cachedDummy;
        }
        if (!xfbDummy) {
            xfbDummy = mglTessCreateBuffer(
                _device, (NSUInteger)dummyBytes, MGL_TESS_RESOURCE_STORAGE_SHARED);
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

    const BOOL indexed = _tessellation.tessIndexedDraw;
    uint32_t gatherVerts = 0u;
    uint32_t gatherPrims = 0u;
    mglTessPlanEvalGather(indexed ? 1 : 0,
                          (uint32_t)_tessellation.tessInstanceRecords,
                          contract->patch_vertices, patchCount, &gatherVerts,
                          &gatherPrims);
    MGLTessEvalPerPatchDispatchSpec patchSpec;
    mglTessFillEvalPerPatchSpec(
        (__bridge void *)glInBuffer, (uint64_t)glInOffset,
        (uint64_t)glInInstanceStride,
        indexed ? (__bridge void *)controlPointIndexBuffer : NULL, gatherVerts,
        gatherPrims, indexed ? 1 : 0, (uint32_t)glInVertices, patchCount,
        instanceCountU, itemsPerInstanceU, &patchSpec);
    void *patchKeepAlive = NULL;
    if (!mglTessAppendEvalPerPatchDispatches(&executionPlan, tesProgram,
                                             factorBytes, &patchSpec,
                                             &patchKeepAlive)) {
        free(patchKeepAlive);
        [self clearStageBindingCopyBacks:&stageCopyBacks];
        return false;
    }
    if (patchKeepAlive) {
        NSData *keep = [[NSData alloc] initWithBytesNoCopy:patchKeepAlive
                                                    length:1
                                               deallocator:^(void *bytes,
                                                             NSUInteger length) {
            (void)length;
            free(bytes);
        }];
        if (!keep) {
            free(patchKeepAlive);
            [self clearStageBindingCopyBacks:&stageCopyBacks];
            return false;
        }
        [executionTemporaries addObject:keep];
    }
    {
        MGLRenderCopyBackEntry copyBackEntries[kMGLMaxBufferSlots] = {0};
        uint32_t copyBackEntryCount = mglRenderCollectCopyBackEntries(
            (const MGLRenderCopyBackEntry *)stageCopyBacks.slots,
            kMGLMaxBufferSlots, copyBackEntries, kMGLMaxBufferSlots);
        executionPlan.barrier_scope = MGL_RENDER_COMPUTE_BARRIER_BUFFERS;
        MGLRenderComputeExecutionResult executionResult = {0};
        char executionError[256] = {0};
        if (mglRenderExecuteComputeExecutionPlan(
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

    if (mglTessXFBCopyBackReady((uint64_t)xfbWrittenBytes,
                                xfbTemporary ? 1 : 0,
                                xfbDestination ? 1 : 0)) {
        const uint8_t *srcBase =
            (const uint8_t *)mglTessBufferContents(xfbTemporary);
        if (!srcBase) {
            NSLog(@"MGL TESS XFB: missing temporary contents");
            return false;
        }
        const bool separateAttribs =
            mglXfbSeparateAttribs(
                tesProgram->transform_feedback_buffer_mode) != 0;
        if (separateAttribs) {
            /* One GL buffer binding per varying (GL 4.6 §11.1.3.2). */
            for (GLsizei varying = 0;
                 varying < tesProgram->transform_feedback_varying_count;
                 varying++) {
                if (!mglXfbVaryingSlotValid((uint32_t)varying)) {
                    break;
                }
                const char *name =
                    tesProgram->transform_feedback_varying_names[varying];
                uint32_t recordOffset = 0u;
                uint32_t fieldType = 0u;
                uint32_t fieldBytes = 0u;
                if (!mglTessResolveXFBSource(tesProgram, name, &recordOffset,
                                             &fieldType, &fieldBytes)) {
                    continue;
                }
                (void)recordOffset;
                (void)fieldType;
                BufferBaseTarget *slot =
                    &MGL_STATE(glm_ctx)
                         ->buffer_base[_TRANSFORM_FEEDBACK_BUFFER]
                         .buffers[varying];
                Buffer *destBuf = slot->buf;
                if (!destBuf) {
                    continue;
                }
                if (mglRenderBufferNeedsCPUUpload(
                        destBuf->size, destBuf->data.dirty_bits)) {
                    if (![self updateDirtyBuffer:destBuf]) {
                        return false;
                    }
                }
                if (!destBuf->data.mtl_data) {
                    [self bindMTLBuffer:destBuf];
                }
                id destMTL = (__bridge id)(destBuf->data.mtl_data);
                const bool sessionOffsetOK =
                    xfbState->buffer_write_offsets[varying] <=
                    (GLuint64)NSUIntegerMax;
                const uint64_t sessionOffset =
                    sessionOffsetOK
                        ? (uint64_t)xfbState->buffer_write_offsets[varying]
                        : 0u;
                uint64_t visible = 0u;
                if (destMTL && slot->offset >= 0) {
                    BufferMap xfbMap = {0};
                    xfbMap.buf = destBuf;
                    xfbMap.offset = slot->offset;
                    xfbMap.size = slot->size;
                    visible = (uint64_t)mglBufferMapVisibleBackingBytes(
                        &xfbMap, (size_t)mglTessBufferLength(destMTL));
                }
                MGLXfbVsBufferDest dest = {0};
                if (!mglXfbPlanVsBufferDestOrUnbacked(
                        (uint32_t)xfbCopiedVertices, fieldBytes,
                        destMTL ? 1 : 0, slot->offset, sessionOffset, visible,
                        &dest) ||
                    dest.skip) {
                    continue;
                }
                NSUInteger destOffset = (NSUInteger)dest.destination_offset;
                NSUInteger maxVerts = dest.written_records;
                NSUInteger written = dest.written_bytes;
                uint8_t *packed = (uint8_t *)calloc(1u, written);
                if (!packed) {
                    NSLog(@"MGL TESS XFB: OOM packing separate attrib %d",
                          (int)varying);
                    return false;
                }
                mglTessPackXFBSeparate(tesProgram, name, srcBase,
                                       (uint32_t)outStride, (uint32_t)maxVerts,
                                       packed);
                mglRendererBufferSubData(glm_ctx, destBuf, (GLintptr)destOffset,
                                         (GLsizeiptr)written, packed);
                if (destMTL) {
                    uint8_t *live =
                        (uint8_t *)mglTessBufferContents(destMTL);
                    if (live) {
                        memcpy(live + destOffset, packed, written);
                    }
                }
                if (mglXfbCPUShadowFits(destBuf->data.buffer_data ? 1 : 0,
                                        destBuf->size, (uint64_t)destOffset,
                                        (uint64_t)written)) {
                    memcpy((uint8_t *)destBuf->data.buffer_data + destOffset,
                           packed, written);
                }
                mglRenderMarkBufferCPUWrite(destBuf, (int64_t)destOffset,
                                            (int64_t)written);
                free(packed);
            }
        } else {
        uint8_t *packed = (uint8_t *)calloc(1u, xfbWrittenBytes);
        if (!packed) {
            NSLog(@"MGL TESS XFB: missing temporary contents or OOM");
            return false;
        }
        mglTessPackXFBInterleaved(tesProgram, srcBase, (uint32_t)outStride,
                                  (uint32_t)xfbCopiedVertices, packed,
                                  (uint32_t)xfbCompactStride);
        mglRendererBufferSubData(glm_ctx, xfbDestination,
                                 xfbCopyDestinationOffset, xfbWrittenBytes,
                                 packed);
        /* Mirror into the live Metal allocation: SubData may land in a
         * snapshot while glMapBufferRange serves the CPU shadow. */
        if (xfbCopyDestination) {
            uint8_t *live = (uint8_t *)mglTessBufferContents(xfbCopyDestination);
            if (live) {
                memcpy(live + xfbCopyDestinationOffset, packed,
                       xfbWrittenBytes);
            }
        }
        if (mglXfbCPUShadowFits(xfbDestination->data.buffer_data ? 1 : 0,
                                xfbDestination->size,
                                (uint64_t)xfbCopyDestinationOffset,
                                (uint64_t)xfbWrittenBytes)) {
            memcpy((uint8_t *)xfbDestination->data.buffer_data +
                       xfbCopyDestinationOffset,
                   packed, xfbWrittenBytes);
        }
        mglRenderMarkBufferCPUWrite(xfbDestination,
                                    (int64_t)xfbCopyDestinationOffset,
                                    (int64_t)xfbWrittenBytes);
        free(packed);
        }
    }
    if (mglXfbShouldAdvanceWriteOffset(xfbActive ? 1 : 0,
                                       (uint64_t)xfbWrittenBytes)) {
        xfbState->buffer_write_offsets[0] = mglXfbAdvanceWriteOffset(
            xfbState->buffer_write_offsets[0], (uint64_t)xfbWrittenBytes);
    }

    /* Rasterize through the passthrough vertex stage, or hand the expanded
     * records to a following geometry shader (coverage VS+TC+TE+GS path). */
    const GLenum tessRasterMode = mglTessRasterGLMode(tesProgram);
    MGLTessRasterQueryPlan query = {0};
    mglTessPlanRasterQuery(tesProgram, (uint64_t)instanceCount,
                           (uint64_t)itemsPerInstanceU, xfbActive ? 1 : 0,
                           (uint64_t)xfbWrittenBytes,
                           (uint32_t)xfbCompactStride, &query);
    MGLTessEvalAfterComputePlan after = {0};
    if (!mglTessPlanEvalAfterCompute(gsProgram ? 1 : 0,
                                     MGL_STATE(glm_ctx)->caps.rasterizer_discard
                                         ? 1
                                         : 0,
                                     itemsPerInstanceU, instanceCountU,
                                     &after)) {
        return false;
    }
    if (after.action == MGL_TESS_AFTER_COMPUTE_GS) {
        if (after.gs_empty) {
            NSLog(@"MGL TESS ERROR: TES→GS empty expansion program=%u",
                  (unsigned)tesProgram->name);
            return false;
        }
        GLsizei gsCount = (GLsizei)after.gs_vertex_count;
        _tessellation.pendingGSInputActive = YES;
        _tessellation.pendingGSInput = (__bridge_retained void *)outBuffer;
        _tessellation.pendingGSInputOffset = 0u;
        _tessellation.pendingGSInputStride = outStride;
        _tessellation.pendingGSVertexCount = gsCount;
        const BOOL gsOK = [self handleGeometryDrawIfNeeded:glm_ctx
                                                      mode:tessRasterMode
                                                     first:0
                                                     count:gsCount
                                                 indexType:0
                                                   indices:NULL
                                                baseVertex:0
                                             instanceCount:1
                                              baseInstance:baseInstance
                                                     label:"tessEvalToGeometry"];
        if (_tessellation.pendingGSInput) {
            (void)CFBridgingRelease(_tessellation.pendingGSInput);
            _tessellation.pendingGSInput = NULL;
        }
        _tessellation.pendingGSInputActive = NO;
        _tessellation.pendingGSInputOffset = 0u;
        _tessellation.pendingGSInputStride = 0u;
        _tessellation.pendingGSVertexCount = 0;
        return gsOK;
    }
    if (after.action == MGL_TESS_AFTER_COMPUTE_DISCARD) {
        /* GL_RASTERIZER_DISCARD: no pixels by definition, so skip the
         * passthrough draw entirely, but the compute expansion already ran
         * and the primitive query must still count the generated
         * primitives (persistent query semantics). */
        _currentCBHasWork = YES;
        mglRecordActivePrimitiveQueryDraw(glm_ctx, query.prims, query.written);
        return YES;
    }
    if (![self ensureAIRTessEvalPassthroughFunctionForProgram:tesProgram]) {
        NSLog(@"MGL TESS ERROR: TES passthrough vertex unavailable program=%u",
              (unsigned)tesProgram->name);
        /* XFB capture already completed above; do not fail the draw and
         * leave transform feedback active when the test only needed feedback. */
        if (mglTessPassthroughFailIsXFBSuccess(xfbActive ? 1 : 0)) {
            mglRecordActivePrimitiveQueryDraw(glm_ctx, query.prims, query.written);
            return YES;
        }
        return false;
    }
    uint32_t primType = mglTessRasterPrimitiveType(tesProgram);

    _tessellation.tessComputeActive = YES;
    _tessellation.tessComputeProgram = tesProgram;
    BOOL stateReady = [self processGLState:true];
    if (!mglTessPassthroughRasterReady(
            stateReady ? 1 : 0,
            mglRenderEncoderOwnerHasCurrent(
                _renderPassManager.state->currentRenderEncoderOwner),
            [self currentDrawRasterizationIsEmpty] ? 1 : 0)) {
        NSLog(@"MGL TESS ERROR: TES compute raster skip program=%u stateReady=%d encoder=%d empty=%d clip0=%d",
              (unsigned)tesProgram->name,
              (int)stateReady,
              mglRenderEncoderOwnerHasCurrent(
                  _renderPassManager.state->currentRenderEncoderOwner),
              (int)[self currentDrawRasterizationIsEmpty],
              ctx && MGL_STATE(ctx)->caps.clip_distances[0] ? 1 : 0);
        _tessellation.tessComputeActive = NO;
        _tessellation.tessComputeProgram = NULL;
        if (mglTessPassthroughFailIsXFBSuccess(xfbActive ? 1 : 0)) {
            mglRecordActivePrimitiveQueryDraw(glm_ctx, query.prims, query.written);
            /* Feedback already landed; returning NO would raise
             * INVALID_OPERATION and skip the test's EndTransformFeedback. */
            return YES;
        }
        return NO;
    }

    [self applyPolygonOffsetForDrawMode:tessRasterMode];
    id encoder = nil;
    for (GLsizei i = 0; i < instanceCount; i++) {
        NSUInteger instanceOffset = (NSUInteger)mglTessPassthroughInstanceOffset(
            (uint32_t)i, itemsPerInstanceU, (uint32_t)outStride);
        mglTessSetRenderVertexBuffer(
            encoder, _renderPassManager.state->currentRenderEncoderOwner,
            outBuffer, instanceOffset, 0u);
        mglTessDrawPrimitives(
            encoder, _renderPassManager.state->currentRenderEncoderOwner,
            primType, 0u, (NSUInteger)itemsPerInstanceU, 1u,
            (NSUInteger)baseInstance + (NSUInteger)i);
    }
    _currentCBHasWork = YES;
    mglRecordActivePrimitiveQueryDraw(glm_ctx, query.prims, query.written);
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
-(bool) dispatchTessEvaluationShader:(GLMContext)glm_ctx
                            program:(Program *)tesProgram
                           contract:(const MGLAIRTessDrawContract *)contract
{
    if (!tesProgram || !glm_ctx || !contract || !tesProgram->tess_eval_compute)
        return false;
    return [self dispatchAIRTessEvalCompute:glm_ctx
                                   program:tesProgram
                                  contract:contract
                                patchCount:contract->patch_count
                             instanceCount:(GLsizei)contract->instance_count
                              baseInstance:contract->base_instance];
}

@end
