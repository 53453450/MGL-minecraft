/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+Compute.m
// Compute dispatch methods extracted from MGLRenderer.m.
// These methods do not depend on any file-scope static functions in MGLRenderer.m.

#import "MGLRenderer_Private.h"
#include "mgl_texture_binding_resolve.h"
#import "MGLRenderer+Binding_Private.h"
#import "mgl_compute_pipeline_cache.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"
#include "mgl_buffer_map.h"  /* buffer mapping entries (was MGLRenderer+Buffer.m) */
#include "mgl_compute_bind.h"  /* compute buffer binding (was a method pair) */
#include "mgl_renderer_ports.h"  /* mglRendererProcessBuffer */
#include "mgl_draw_tess.h"

enum {
    MGL_COMPUTE_TEXTURE_TYPE_CUBE = 5u,
    MGL_COMPUTE_TEXTURE_TYPE_CUBE_ARRAY = 6u,
};

static id mglComputeCreateBufferWithBytes(
    const void *bytes,
    NSUInteger length,
    uint64_t resourceOptions)
{
    void *buffer = NULL;
    if (mglRenderCreateBufferWithBytes(bytes, length, resourceOptions, NULL,
                                          &buffer) == 0 && buffer) {
        return (__bridge_transfer id)buffer;
    }
    return nil;
}

static id mglComputeCreateDefaultSampler(void)
{
    void *sampler = NULL;
    if (mglRenderCreateDefaultSampler(&sampler) == 0 && sampler) {
        return (__bridge_transfer id)sampler;
    }
    return nil;
}

static void mglComputeSetBuffer(id encoder,
                                id buffer,
                                NSUInteger offset,
                                NSUInteger index)
{
    (void)mglRenderSetComputeBuffer(
        (__bridge void *)encoder, (__bridge void *)buffer,
        (uint64_t)offset, (uint32_t)index);
}

static void mglComputeSetTexture(id encoder,
                                 id texture,
                                 NSUInteger index)
{
    (void)mglRenderSetComputeTexture(
        (__bridge void *)encoder, (__bridge void *)texture,
        (uint32_t)index);
}

static void mglComputeSetSampler(id encoder,
                                 id sampler,
                                 NSUInteger index)
{
    (void)mglRenderSetComputeSampler(
        (__bridge void *)encoder, (__bridge void *)sampler,
        (uint32_t)index);
}

static void mglComputeSetPipeline(id encoder, id pipeline)
{
    (void)mglRenderSetComputePipelineState(
        (__bridge void *)encoder, (__bridge void *)pipeline);
}

static void mglComputeDispatch(id encoder,
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

static void mglComputeDispatchIndirect(id encoder,
                                       id buffer,
                                       NSUInteger offset,
                                       uint32_t threadsX,
                                       uint32_t threadsY,
                                       uint32_t threadsZ)
{
    (void)mglRenderDispatchComputeIndirect(
        (__bridge void *)encoder, (__bridge void *)buffer,
        (uint64_t)offset, threadsX, threadsY, threadsZ);
}

static void mglComputeEndEncoder(id encoder)
{
    (void)mglRenderEndComputeEncoder((__bridge void *)encoder);
}

@interface MGLRenderer (ComputeLocked)
- (void)mtlDispatchComputeLocked:(GLMContext)glm_ctx
                         groupsX:(GLuint)groups_x
                         groupsY:(GLuint)groups_y
                         groupsZ:(GLuint)groups_z;
- (void)mtlDispatchComputeIndirectLocked:(GLMContext)glm_ctx
                                indirect:(GLintptr)indirect;
@end

void mglRendererDispatchCompute(GLMContext glm_ctx,
                                      unsigned int groups_x,
                                      unsigned int groups_y,
                                      unsigned int groups_z)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        METAL_LOCK();
        [renderer mtlDispatchComputeLocked:glm_ctx
                                   groupsX:groups_x
                                   groupsY:groups_y
                                   groupsZ:groups_z];
        METAL_UNLOCK();
    }
    mglRendererBackendEnd(&_backend_lease);
}

void mglRendererDispatchComputeIndirect(GLMContext glm_ctx,
                                              intptr_t indirect)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx) {
        METAL_LOCK();
        [renderer mtlDispatchComputeIndirectLocked:glm_ctx indirect:indirect];
        METAL_UNLOCK();
    }
    mglRendererBackendEnd(&_backend_lease);
}

@implementation MGLRenderer (Compute)

#pragma mark ----- compute utility ---------------------------------------------------------------------

- (bool) bindTexturesToComputeEncoder:(id) computeCommandEncoder
                                 stage:(int)stage
{
    return [self bindTexturesToComputeEncoder:computeCommandEncoder
                                         stage:stage
                                 executionPlan:NULL
                                  temporaries:nil];
}

- (bool) bindTexturesToComputeEncoder:(id) computeCommandEncoder
                                 stage:(int)stage
                         executionPlan:(MGLRenderComputeExecutionPlan *)executionPlan
                          temporaries:(NSMutableArray *)temporaries
{
    GLuint count;
    if (!computeCommandEncoder && !executionPlan) {
        NSLog(@"MGL COMPUTE ERROR: NULL compute encoder for texture binding");
        return false;
    }


    const BOOL useComputeTextureSnapshot = YES;
    BOOL textureSnapshotOK = YES;
    MGLRenderComputeBindingSnapshot ctexSnapshot = {0};
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
                if (mglRenderAppendComputeBindingSnapshotToPlan(             \
                        executionPlan, &ctexSnapshot, NULL, 0) != 0) {          \
                    textureSnapshotOK = NO;                                     \
                }                                                               \
            } else {                                                            \
                textureSnapshotOK = mglRenderEncodeComputeBindingSnapshot(    \
                    (__bridge void *)computeCommandEncoder, &ctexSnapshot,     \
                    NULL, 0) == 0 && textureSnapshotOK;                          \
            }                                                                   \
            ctexSnapshot = (MGLRenderComputeBindingSnapshot){0};             \
        }                                                                       \
    } while (0)

#define MGL_CTEX_EMIT_TEXTURE(slot, texPtr)                                     \
    do {                                                                        \
        if (useComputeTextureSnapshot) {                                        \
            if (ctexSnapshot.op_count >=                                        \
                MGL_RENDER_COMPUTE_BINDING_SNAPSHOT_MAX_OPS) {              \
                MGL_CTEX_FLUSH_SNAPSHOT();                                      \
            }                                                                   \
            ctexSnapshot.ops[ctexSnapshot.op_count++] =                         \
                (MGLRenderComputeBindingOp){/* kind */ 2u,                   \
                                               /* index */ (uint32_t)(slot),    \
                                               /* offset */ 0,                  \
                                               /* buffer */ (void *)(texPtr),   \
                                               /* bytes */ NULL,                \
                                               /* length */ 0u};                \
        } else {                                                                \
            mglComputeSetTexture(computeCommandEncoder,                         \
                                 (__bridge id)(texPtr), (slot));                \
        }                                                                       \
    } while (0)

#define MGL_CTEX_EMIT_SAMPLER(slot, smpPtr)                                     \
    do {                                                                        \
        if (useComputeTextureSnapshot) {                                        \
            if (ctexSnapshot.op_count >=                                        \
                MGL_RENDER_COMPUTE_BINDING_SNAPSHOT_MAX_OPS) {              \
                MGL_CTEX_FLUSH_SNAPSHOT();                                      \
            }                                                                   \
            ctexSnapshot.ops[ctexSnapshot.op_count++] =                         \
                (MGLRenderComputeBindingOp){/* kind */ 3u,                   \
                                               /* index */ (uint32_t)(slot),    \
                                               /* offset */ 0,                  \
                                               /* buffer */ (void *)(smpPtr),   \
                                               /* bytes */ NULL,                \
                                               /* length */ 0u};                \
        } else {                                                                \
            mglComputeSetSampler(computeCommandEncoder,                         \
                                 (__bridge id)(smpPtr),                         \
                                 (slot));                                       \
        }                                                                       \
    } while (0)

    Program *computeProgram = mglResolveProgramForStageFromState(ctx, stage);

    static const int kComputeTextureSpvcTypes[] = {
        _SAMPLED_IMAGE_RES, _STORAGE_IMAGE_RES,
    };
    for (int type = 0; type < 2; type++)
    {
        int spvc_type = kComputeTextureSpvcTypes[type];
        int gl_texture_type =
            mglRenderComputeTextureBindKind((uint32_t)spvc_type);
        if (gl_texture_type < 0) {
            continue;
        }

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
                GLuint resourceElement = 0u;
                GLuint metalBinding = mglRendererGetProgramBinding(ctx, stage, spvc_type, i);
                GLuint glUnit = 0u;
                Texture *ptr = NULL;

                if (computeProgram &&
                    spvc_type >= 0 && spvc_type < MGL_MAX_SHADER_RESOURCES &&
                    i >= 0) {
                    MGLShaderResourceList *resourceList =
                        &computeProgram->shader_resources_list[stage][spvc_type];
                    if (mglRenderComputeTextureListExpandsByElement(
                            (uint32_t)spvc_type)) {
                        GLuint ordinal = (GLuint)i;
                        for (GLuint ri = 0; ri < resourceList->count; ri++) {
                            MGLShaderResource *candidate = &resourceList->list[ri];
                            GLuint elements = mglRenderShaderResourceElementCount(
                                (uint32_t)candidate->gl_array_size);
                            if (ordinal < elements) {
                                resource = candidate;
                                resourceElement = ordinal;
                                metalBinding = candidate->binding + ordinal;
                                break;
                            }
                            ordinal -= elements;
                        }
                    } else if (i < (int)resourceList->count) {
                        resource = &resourceList->list[i];
                        metalBinding = mglMetalResourceSlot(resource);
                    }
                }

                if (mglRenderMetalBindingPastUnits(metalBinding, TEXTURE_UNITS)) {
                    continue;
                }

                if (mglRenderComputeTextureBindIsStorage(
                        (uint32_t)gl_texture_type))
                {
                        const int explicitUnit =
                            computeProgram && metalBinding < TEXTURE_UNITS &&
                            computeProgram->sampler_units_explicit_by_stage[stage]
                                                                          [metalBinding];
                        if (explicitUnit || resource) {
                            glUnit = mglRenderImageUnitFromResource(
                                explicitUnit,
                                explicitUnit
                                    ? (uint32_t)computeProgram
                                          ->sampler_units_by_stage[stage]
                                                                  [metalBinding]
                                    : 0u,
                                resource ? resource->sampler_unit : -1,
                                resource ? resource->gl_binding : 0u,
                                resourceElement);
                        } else {
                            glUnit = (GLuint)mglRendererGetProgramGLBinding(
                                ctx, stage, spvc_type, i);
                        }
                        if (!mglRenderImageUnitsInRange(0u, glUnit,
                                                        TEXTURE_UNITS)) {
                            continue;
                        }
                        ptr = MGL_STATE(ctx)->image_units[glUnit].tex;
                } else {
                        glUnit = mglTextureUnitForSampledResource(resource, mglResolveProgramForStageFromState(ctx, stage), metalBinding, stage);
                        if (glUnit >= TEXTURE_UNITS) {
                            continue;
                        }
                        ptr = mglTextureForSampledResourceForStage(ctx, resource, metalBinding, stage, mglRendererGetProgramDeclaredTextureType(
                                                      ctx, stage, spvc_type, i));
                }

                if (ptr)
                {
                    RETURN_FALSE_ON_FAILURE([self bindMTLTexture: ptr]);
                    if (!ptr->mtl_data) {
                        continue;
                    }

                    id texture = (__bridge id)(ptr->mtl_data);
                    if (!texture) {
                        continue;
                    }

                    /* Storage images: BindImage <format>/level/slice views
                     * (same helper as VS/FS). Cached on ImageUnit. */
                    if (mglRenderComputeTextureBindIsStorage(
                            (uint32_t)gl_texture_type)) {
                        texture = (__bridge id)mglRendererStorageImageTexture(
                            (__bridge void *)texture,
                            &MGL_STATE(ctx)->image_units[glUnit]);
                    }

                    /* Sampler cascade (GL sampler object → texture parameters
                     * → default) is the shared materialize port the
                     * vertex / fragment spine uses. */
                    id sampler = [self
                        materializeSampledSamplerForTexture:ptr
                                                textureUnit:glUnit
                                            defaultSampler:nil
                                              forceDefault:NO
                                             samplerTarget:ptr->target
                                               programName:computeProgram
                                                               ? computeProgram->name
                                                               : 0u
                                              spirvBinding:resource
                                                               ? mglMetalResourceSlot(resource)
                                                               : metalBinding
                                                     stage:"compute"
                                                   texture:texture];

                    if (!sampler) {
                        id fallbackSampler = mglComputeCreateDefaultSampler();
                        sampler = fallbackSampler;
                        /* Keep the fallback alive until the end replay. */
                        MGL_CTEX_RETAIN_TEMP(sampler);
                        if (!sampler) {
                            continue;
                        }
                    }

                    MGL_CTEX_EMIT_TEXTURE(metalBinding,
                                          (__bridge void *)texture);
                    if (mglRenderComputeTextureBindNeedsSampler(
                            (uint32_t)gl_texture_type,
                            !resource || resource->has_combined_sampler)) {
                        GLuint samplerBinding = resource
                            ? mglMetalCombinedSamplerSlotForElement(resource,
                                                                    resourceElement)
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

            uint32_t expectedType =
                mglRendererGetProgramDeclaredTextureType(ctx, stage, _SAMPLED_IMAGE_RES, (int)resourceIndex);
            for (GLint element = 1; element < resource->gl_array_size; element++) {
                GLuint metalSlot = resource->binding + (GLuint)element;
                GLuint samplerSlot =
                    mglMetalCombinedSamplerSlotForElement(resource,
                                                          (GLuint)element);
                if (metalSlot >= TEXTURE_UNITS) {
                    break;
                }

                GLuint glUnit = mglTextureUnitForSampledResource(NULL, mglResolveProgramForStageFromState(ctx, stage), metalSlot, stage);
                Texture *ptr = mglTextureForSampledResourceForStage(ctx, NULL, metalSlot, stage, expectedType);
                if (!ptr || ![self bindMTLTexture:ptr] || !ptr->mtl_data) {
                    continue;
                }

                id texture = (__bridge id)(ptr->mtl_data);
                /* Same shared port as the loop above.  This path used to
                 * skip the "dirty sampler" release the others do, so a
                 * re-parameterized sampler could keep its old Metal object;
                 * going through the port makes it consistent. */
                id sampler = [self
                    materializeSampledSamplerForTexture:ptr
                                            textureUnit:glUnit
                                        defaultSampler:nil
                                          forceDefault:NO
                                         samplerTarget:ptr->target
                                           programName:computeProgram
                                                           ? computeProgram->name
                                                           : 0u
                                          spirvBinding:metalSlot
                                                 stage:"compute"
                                               texture:texture];

                if (!sampler) {
                    sampler = mglComputeCreateDefaultSampler();
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
- (bool)processCompute:(id)computeCommandEncoder
             copyBacks:(MGLStageBindingCopyBackList *)copyBacks
{
    return [self processCompute:computeCommandEncoder
                       copyBacks:copyBacks
                   executionPlan:NULL
                    temporaries:nil];
}

- (bool)processCompute:(id)computeCommandEncoder
             copyBacks:(MGLStageBindingCopyBackList *)copyBacks
         executionPlan:(MGLRenderComputeExecutionPlan *)executionPlan
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
        mglDispatchError(ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
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
        mglDispatchError(ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return false;
    }

    id func = (__bridge id)(program->modules[_COMPUTE_SHADER].mtl_function);
    if (!func) {
        NSLog(@"MGL COMPUTE ERROR: compute shader for program %u has no Metal function", program->name);
        return false;
    }

    void *computePipelineHandle = NULL;
    char computePipelineError[2048] = {0};
    int computePipelineResult = mglGetOrCreateProgramComputePipeline(
        program, _COMPUTE_SHADER, &computePipelineHandle,
        computePipelineError, sizeof(computePipelineError));
    id computePipelineState =
        computePipelineResult == 0 && computePipelineHandle
            ? (__bridge_transfer id)computePipelineHandle
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

    RETURN_FALSE_ON_FAILURE(mglComputeBindBuffersToEncoder(
        (__bridge void *)self, _COMPUTE_SHADER,
        (__bridge void *)computeCommandEncoder, copyBacks, executionPlan,
        (__bridge void *)temporaries));

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



- (BOOL)runComputeDispatchOrchestrationLocked:(GLMContext)glm_ctx
                                  dispatchKind:(uint32_t)dispatchKind
                                     groupsX:(GLuint)groups_x
                                     groupsY:(GLuint)groups_y
                                     groupsZ:(GLuint)groups_z
                              indirectBuffer:(id)indirectBuffer
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
    const BOOL useExecutionPlan = YES;
    MGLRenderComputeExecutionPlan executionPlan = {0};
    NSMutableArray *executionTemporaries = useExecutionPlan
        ? [NSMutableArray array] : nil;
    id computeCommandEncoder = nil;
    if (!useExecutionPlan) {
        computeCommandEncoder =
            (__bridge id)mglRenderCreateComputeEncoderBorrowed(
                _renderPassManager->state->currentCommandBufferOwner);
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
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return NO;
    }


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
            ? MGL_RENDER_COMPUTE_BARRIER_BUFFERS
            : MGL_RENDER_COMPUTE_BARRIER_NONE;
        executionPlan.dispatch = (MGLRenderComputePlan){
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
        MGLRenderCopyBackEntry copyBackEntries[kMGLMaxBufferSlots] = {0};
        uint32_t copyBackEntryCount = mglRenderCollectCopyBackEntries(
            (const MGLRenderCopyBackEntry *)copyBacks.slots, kMGLMaxBufferSlots,
            copyBackEntries, kMGLMaxBufferSlots);
        MGLRenderComputeExecutionResult executionResult = {0};
        char executionError[256] = {0};
        if (mglRenderExecuteComputeExecutionPlan(
                _renderPassManager->state->currentCommandBufferOwner,
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
            mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
            return NO;
        }
        [self clearStageBindingCopyBacks:&copyBacks];
    } else {

        MGLRenderThreadgroupSize tg = {0};
        mglRenderThreadgroupSize(
            ptr->local_workgroup_size.x, ptr->local_workgroup_size.y,
            ptr->local_workgroup_size.z, &tg);
        if (dispatchKind == MGL_RENDER_COMPUTE_DISPATCH_DIRECT) {
            mglComputeDispatch(computeCommandEncoder,
                               groups_x, groups_y, groups_z,
                               tg.x, tg.y, tg.z);
        } else {
            mglComputeDispatchIndirect(computeCommandEncoder, indirectBuffer,
                                       indirectOffset, tg.x, tg.y, tg.z);
        }
    }

    if (computeCommandEncoder) {
        mglComputeEndEncoder(computeCommandEncoder);
    }
    /* Without this, a dispatch with no copy-backs stays in the current
     * command buffer and flushCommandBufferLocked's empty-CB skip drops it:
     * glFinish then never executes the compute writes (SSBO stores vanish). */
    _batching.currentCommandBufferHasWork = YES;

    if (!useExecutionPlan &&
        ![self flushStageBindingCopyBacks:&copyBacks
                     requireCPUVisibility:NO]) {
        NSLog(@"MGL COMPUTE ERROR: failed to copy isolated writable buffer prefixes after %s",
              reason ? reason : "dispatch");
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorOutOfMemory());
        return NO;
    }
    if (useExecutionPlan && hasCopyBackEntries &&
        ![self newCommandBufferLocked]) {
        NSLog(@"MGL COMPUTE ERROR: failed to install post-compute command buffer after %s",
              reason ? reason : "dispatch");
        return NO;
    }


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


    if (![self runComputeDispatchOrchestrationLocked:glm_ctx
                                        dispatchKind:MGL_RENDER_COMPUTE_DISPATCH_DIRECT
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
            !mglRenderImageAccessWritable((uint32_t)imageUnit->access)) {
            continue;
        }
        imageTexture->metal_data_authoritative = (GLboolean)mglRenderGLBoolean(1);
        if (imageTexture->faces[0].levels &&
            imageUnit->level >= 0 &&
            imageUnit->level < (GLint)imageTexture->num_levels) {
            imageTexture->faces[0].levels[imageUnit->level].metal_data_authoritative = (GLboolean)mglRenderGLBoolean(1);
        }
    }

    //[self newRenderEncoder];
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
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return;
    }
    if (indirect < 0) {
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidValue());
        return;
    }

    if (!mglRendererProcessBuffer((__bridge void *)self, glIndirectBuffer)) {
        NSLog(@"MGL COMPUTE ERROR: failed to process dispatch indirect buffer %u",
              glIndirectBuffer ? glIndirectBuffer->name : 0u);
        return;
    }

    id indirectBuffer = (__bridge id)(glIndirectBuffer->data.mtl_data);
    if (!indirectBuffer) {
        NSLog(@"MGL COMPUTE ERROR: dispatch indirect buffer %u has no Metal backing",
              glIndirectBuffer ? glIndirectBuffer->name : 0u);
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    NSUInteger indirectOffset = (NSUInteger)indirect;
    NSUInteger indirectArgBytes = 3u * sizeof(uint32_t);
    MGLRenderBufferInfo indirectBufferInfo = {0};
    if (mglRenderGetBufferInfo((__bridge void *)indirectBuffer,
                                  &indirectBufferInfo) != 0 ||
        indirectOffset > indirectBufferInfo.length ||
        indirectArgBytes > (indirectBufferInfo.length - indirectOffset)) {
        NSLog(@"MGL COMPUTE ERROR: dispatch indirect range exceeds Metal buffer buffer=%u off=%lu bytes=%lu len=%lu",
              glIndirectBuffer ? glIndirectBuffer->name : 0u,
              (unsigned long)indirectOffset,
              (unsigned long)indirectArgBytes,
              (unsigned long)indirectBufferInfo.length);
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return;
    }


    if (![self runComputeDispatchOrchestrationLocked:glm_ctx
                                        dispatchKind:MGL_RENDER_COMPUTE_DISPATCH_INDIRECT
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
