/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_compute_bind.c — the buffer half of MGLRenderer+Compute.m moved here
 * (P0-1, log 109).  The Objective-C macros became static functions over a small
 * context struct:
 *
 *   MGL_CBIND_FLUSH_SNAPSHOT()   -> mglComputeBindFlushSnapshot()
 *   MGL_CBIND_RETAIN_TEMP(obj)   -> mglComputeBindRetainTemp()
 *   MGL_CBIND_EMIT_BUFFER(...)   -> mglComputeBindEmitBuffer()
 *
 * `useComputeBindingSnapshot` was the constant YES, so the direct
 * mglComputeSetBuffer path stays as the (dead, as before) else branch and the
 * snapshot path is what runs.  Everything else is the mechanical translation:
 * ctx -> areas.ctx, MGL_STATE(ctx) -> the twin below, "[self ...]" -> the C
 * entries of mgl_renderer_ports.h, NSLog -> fprintf on the same sink.
 */

#include <stdio.h>
#include <string.h>

#include <CoreFoundation/CoreFoundation.h>

#include "mgl_compute_bind.h"
#include "mgl_texture_bind.h"        /* mglRendererBindMTLTexture */
#include "mgl_renderer_ports.h"     /* state areas + the host entries */
#include "mgl_renderer_backend.h"   /* program binding sizes */
#include "mgl_binding_policy.h"
#include "mgl_binding_stage.h"      /* bind plan for a map entry */
#include "mgl_buffer_slots.h"       /* kMGLMaxMetalVertexBufferCount */
#include "mgl_buffer_map.h"         /* map/update-dirty entries */
#include "mgl_draw_tess.h"          /* mglTessRequiredBindingBytes */
#include "mgl_texture_binding_resolve.h" /* sampled-resource lookups */
#include "mgl_texture_compat.h"    /* mglTextureUnitForSampledResource */
#include "mgl_shader_resource.h"   /* mglMetalResourceSlot */
#include "mgl_env_flag.h"

/* MGL_STATE() from MGLRenderer_Private.h, in C. */
static GLMState *mglComputeBindState(const MGLRendererStateAreas *areas)
{
    if (areas->core && areas->core->activeState) {
        return areas->core->activeState;
    }
    return areas->ctx ? areas->ctx->active_state : NULL;
}

/* +1 buffer, or NULL. */
static void *mglComputeBindCreateBufferWithBytes(const void *bytes,
                                                 size_t length,
                                                 uint64_t resourceOptions)
{
    void *buffer = NULL;
    if (mglRenderCreateBufferWithBytes(bytes, length, resourceOptions, NULL,
                                       &buffer) == 0 && buffer) {
        return buffer;
    }
    return NULL;
}

/* The binding encode is batched into a snapshot that is either appended to the
 * execution plan or encoded right away; both paths carry the same op list. */
typedef struct MGLComputeBindCtx_t {
    void *encoder;
    MGLRenderComputeExecutionPlan *plan;
    void *temporaries;
    MGLRenderComputeBindingSnapshot snapshot;
    bool snapshot_ok;
} MGLComputeBindCtx;

static void mglComputeBindFlushSnapshot(MGLComputeBindCtx *ctx)
{
    if (ctx->snapshot.op_count == 0) {
        return;
    }
    if (ctx->plan) {
        if (mglRenderAppendComputeBindingSnapshotToPlan(
                ctx->plan, &ctx->snapshot, NULL, 0) != 0) {
            ctx->snapshot_ok = false;
        }
    } else {
        ctx->snapshot_ok =
            mglRenderEncodeComputeBindingSnapshot(ctx->encoder, &ctx->snapshot,
                                                  NULL, 0) == 0 &&
            ctx->snapshot_ok;
    }
    ctx->snapshot = (MGLRenderComputeBindingSnapshot){0};
}

/* Keeps a temporary alive for the plan's lifetime.  With no plan (the direct
 * encode path) the encoder retains what it binds, so nothing is kept here. */
static void mglComputeBindRetainTemp(MGLComputeBindCtx *ctx, void *object)
{
    if (ctx->plan && ctx->temporaries && object) {
        mglRendererTemporariesAdd(ctx->temporaries, object);
    }
}

/* Hands the created buffer to the keep-alive set and drops the reference this
 * function took: exactly the pair ARC performed (the array retains, the strong
 * local releases at scope exit). */
static void mglComputeBindHandOffCreated(MGLComputeBindCtx *ctx, void *object)
{
    if (!object) {
        return;
    }
    mglComputeBindRetainTemp(ctx, object);
    CFRelease((CFTypeRef)object);
}

static void mglComputeBindEmitBuffer(MGLComputeBindCtx *ctx, uint32_t slot,
                                     void *buffer, uint64_t offset)
{
    if (ctx->snapshot.op_count >= MGL_RENDER_COMPUTE_BINDING_SNAPSHOT_MAX_OPS) {
        mglComputeBindFlushSnapshot(ctx);
    }
    ctx->snapshot.ops[ctx->snapshot.op_count++] =
        (MGLRenderComputeBindingOp){/* kind */ 0u,
                                    /* index */ slot,
                                    /* offset */ offset,
                                    /* buffer */ buffer,
                                    /* bytes */ NULL,
                                    /* length */ 0u};
}

bool mglComputeBindBuffersToEncoder(void *renderer, int stage, void *encoder,
                                    MGLStageBindingCopyBackList *copy_backs,
                                    MGLRenderComputeExecutionPlan *plan,
                                    void *temporaries)
{
    if ((!encoder && !plan) || !copy_backs) {
        fprintf(stderr,
                "MGL COMPUTE ERROR: NULL compute encoder for buffer binding\n");
        return false;
    }

    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    GLMState *state = mglComputeBindState(&areas);

    MGLComputeBindCtx bind = {
        .encoder = encoder,
        .plan = plan,
        .temporaries = temporaries,
        .snapshot_ok = true,
    };

    BufferMapList localBufferMap = {0};
    BufferMapList *bufferMap = mglRenderStageUsesComputeBufferMap(stage)
        ? &state->compute_buffer_map_list : &localBufferMap;
    RETURN_FALSE_ON_FAILURE(
        mglRendererMapGLBuffersToMTLBufferMap(renderer, bufferMap, stage));

    /* dirty buffer covers all buffer modifications */
    if (mglRenderHasDirtyBufferBit(state->dirty_bits))
    {
        /* updateDirtyBaseBufferList binds new mtl buffers or updates old ones */
        (void)mglRendererUpdateDirtyBaseBufferList(renderer, bufferMap);

        state->dirty_bits &= ~DIRTY_BUFFER;
    }

    for (int i = 0; i < bufferMap->count; i++)
    {
        BufferMap *map = &bufferMap->buffers[i];
        Buffer *ptr = map->buf;

        if (!ptr) {
            mglComputeBindFlushSnapshot(&bind);
            fprintf(stderr, "MGL COMPUTE ERROR: buffer map[%d] NULL buffer\n", i);
            return false;
        }

        /* One shared plan decides the Metal slot, the required size, whether
         * the map needs an isolated copy and at which offset to bind
         * (mgl_binding_stage.h).  This loop only materializes what the plan
         * asks for and records the encode.  The three switches below describe
         * the compute stage: it has no set*Bytes path (so a plain-uniform slot
         * must become a real Metal buffer instead of an inline binding) and,
         * as the writer of its own buffers, it isolates a map whose GL storage
         * is exhausted or whose visible backing is empty. */
        size_t requiredBytes = (size_t)mglRendererGetProgramBindingRequiredSize(
            ctx, stage, (int)map->resource_type, (int)map->resource_index);
        requiredBytes = (size_t)mglTessRequiredBindingBytes(
            (int)map->resource_type, (uint32_t)requiredBytes);

        MGLStageBufferBindInput bin = {0};
        mglBindingStageFillMapEntryInput(
            &bin, /*is_fragment=*/0, MGL_SB_PHASE_PRE_MTL,
            /*is_base_binding=*/1, map->has_metal_binding ? 1 : 0,
            map->has_metal_binding ? (int32_t)map->metal_binding_index : -1,
            (int32_t)map->buffer_base_index, (uint32_t)map->resource_type,
            map->offset, ptr->size, /*has_buffer=*/1,
            ptr->data.buffer_data ? 1 : 0, ptr->data.mtl_data ? 1 : 0,
            (const void *)(uintptr_t)ptr->data.buffer_data, ptr->data.mtl_data,
            /*cpu_dirty=*/0, ptr->gpu_write_target ? 1 : 0,
            /*allow_isolate_when_gpu=*/1, /*attrib_reserved=*/0,
            (uint32_t)kMGLMaxMetalVertexBufferCount,
            (uint32_t)MAX_BINDABLE_BUFFERS, (uint32_t)requiredBytes,
            /*min_stage_bytes=*/0u, /*scratch_cap=*/0u,
            /*visible_cpu=*/0u, /*visible_range=*/0);
        bin.no_inline = 1;
        bin.iso_storage_exhausted = 1;
        bin.iso_empty_visible = 1;
        bin.storage_remaining = (int64_t)mglBufferMapStorageRemaining(map);

        MGLStageBufferBindPlan plan_out = {0};
        if (mglBindingStagePlanMapEntry(&bin, &plan_out) != 0) {
            mglComputeBindFlushSnapshot(&bind);
            fprintf(stderr,
                    "MGL COMPUTE ERROR: buffer map[%d] bind plan failed\n", i);
            return false;
        }
        /* An unusable map is refused loudly rather than encoded as a nil
         * binding; only an entry with nothing to bind is skipped, as before.
         * The reason table lives in the plan layer. */
        const int disposition =
            mglBindingStageMapEntryDisposition(plan_out.reason);
        if (disposition != 0) {
            mglComputeBindFlushSnapshot(&bind);
            if (disposition > 0) {
                fprintf(stderr, "MGL COMPUTE WARNING: buffer map[%d] %s, skipping\n",
                        i, mglBindingStagePlanReasonName(plan_out.reason));
                continue;
            }
            fprintf(stderr,
                    "MGL COMPUTE ERROR: buffer map[%d] %s (offset=%lld size=%lld)\n",
                    i, mglBindingStagePlanReasonName(plan_out.reason),
                    (long long)map->offset, (long long)ptr->size);
            return false;
        }
        if (plan_out.action != MGL_SB_ACTION_NEED_MTL) {
            /* No inline path is allowed above, so anything else here means the
             * map cannot be bound. */
            mglComputeBindFlushSnapshot(&bind);
            fprintf(stderr,
                    "MGL COMPUTE ERROR: buffer map[%d] %s is not bindable\n",
                    i, mglBindingStagePlanReasonName(plan_out.reason));
            return false;
        }

        size_t metalBindingIndex = (size_t)plan_out.metal_slot;
        mglRendererClearStageBindingCopyBackPort(renderer, copy_backs,
                                                 (uint64_t)metalBindingIndex);

        /* ---- materialize the Metal backing the plan asked for ---- */
        if (ptr->data.mtl_data) {
            MGLRenderBufferInfo existingInfo = {0};
            if (mglRenderGetBufferInfo(ptr->data.mtl_data, &existingInfo) == 0 &&
                mglRenderMetalBackingTooSmall(ptr->size, existingInfo.length)) {
                /* A plain-uniform buffer may grow after another stage has
                 * materialized a short backing store.  The dirty-update path
                 * preserves the old Metal allocation, so drop it here and
                 * let bindMTLBuffer recreate it at the new GL size. */
                mglRenderReleaseBufferMetalData(ctx, ptr);
            }
        }
        if (!ptr->data.mtl_data) {
            mglRendererBindMTLBuffer(renderer, ptr);
        }
        if (mglRenderBufferHasCPUDirty(ptr->data.dirty_bits)) {
            /* Push pending CPU shadow into Metal.  bindMTLBuffer alone does
             * not clear dirty_bits; leaving them set lets a later VBO bind
             * CoW-overlay the CPU shadow and wipe shader SSBO stores
             * (CTS advanced-write-geometry). */
            if (!mglRendererUpdateDirtyBuffer(renderer, ptr)) {
                fprintf(stderr,
                        "MGL COMPUTE ERROR: dirty buffer update failed buffer=%u\n",
                        (unsigned)ptr->name);
                mglComputeBindFlushSnapshot(&bind);
                return false;
            }
        }
        /* After this encode, Metal is authoritative for writable SSBOs /
         * atomics — drop CPU written_min/max so a later dirty CoW cannot
         * re-apply stale BufferData zeros over GPU stores. */
        if (mglRenderWritableStorageNeedsGPUAuthoritative(
                (int)map->resource_type)) {
            mglRenderClearCPUWriteRange(ptr);
        }

        void *buffer = ptr->data.mtl_data;
        MGLRenderBufferInfo bufferInfo = {0};
        const bool hasBufferInfo = buffer &&
            mglRenderGetBufferInfo(buffer, &bufferInfo) == 0;
        size_t availableBytes = hasBufferInfo
            ? mglBufferMapVisibleBackingBytes(map, bufferInfo.length)
            : 0u;

        mglBindingStageFillMapEntryPostMtl(
            &bin, ptr->data.mtl_data ? 1 : 0, ptr->data.mtl_data,
            hasBufferInfo ? 1 : 0, hasBufferInfo ? bufferInfo.length : 0u,
            (uint64_t)availableBytes, /*binding_state_valid=*/0,
            /*buffer_matches=*/0);
        if (mglBindingStagePlanMapEntry(&bin, &plan_out) != 0) {
            mglComputeBindFlushSnapshot(&bind);
            fprintf(stderr,
                    "MGL COMPUTE ERROR: buffer map[%d] post-ensure plan failed\n",
                    i);
            return false;
        }

        if (plan_out.action == MGL_SB_ACTION_ISOLATE) {
            size_t fallbackLength =
                (size_t)mglBindingStageIsolateFallbackLength(
                    plan_out.required_bytes);
            void *isolated = mglRendererIsolatedStageBindingBufferPort(
                renderer, map, buffer, (uint64_t)fallbackLength);
            if (!isolated) {
                fprintf(stderr,
                        "MGL COMPUTE ERROR: failed to isolate undersized buffer map[%d] buffer=%u required=%lu available=%lu\n",
                        i,
                        (unsigned)ptr->name,
                        (unsigned long)fallbackLength,
                        (unsigned long)availableBytes);
                mglComputeBindFlushSnapshot(&bind);
                return false;
            }

            /* The copy-back writes where the caller's binding starts, not at
             * the plan's bind offset (an isolated binding is always @0). */
            if (plan_out.needs_copy_back &&
                !mglRendererRecordStageBindingCopyBackPort(
                    renderer, copy_backs, (uint64_t)metalBindingIndex,
                    isolated, buffer, ptr, (uint64_t)map->offset,
                    (uint64_t)availableBytes)) {
                mglComputeBindHandOffCreated(&bind, isolated);
                return false;
            }

            /* Isolate the undefined suffix from page-alignment bytes. A
             * post-dispatch blit preserves writes to the legal prefix. */
            mglComputeBindEmitBuffer(&bind, (uint32_t)metalBindingIndex, isolated,
                                     0);
            mglComputeBindHandOffCreated(&bind, isolated);
            /* Isolated buffers are owned only by this loop local (created via
             * __bridge_transfer on gate-on): flush immediately so the encoder
             * retains the buffer while it is still alive, instead of holding a
             * dangling pointer in the snapshot until the end-of-function
             * replay. */
            mglComputeBindFlushSnapshot(&bind);
            continue;
        }

        if (plan_out.action != MGL_SB_ACTION_BIND_BUFFER) {
            mglComputeBindFlushSnapshot(&bind);
            fprintf(stderr,
                    "MGL COMPUTE ERROR: buffer map[%d] not bindable after ensure (%s)\n",
                    i, mglBindingStagePlanReasonName(plan_out.reason));
            return false;
        }

        mglComputeBindEmitBuffer(&bind, (uint32_t)metalBindingIndex, buffer,
                                 plan_out.bind_offset);
        mglNoteBufferEncoded(ptr);
    }

    /* Bind spvBufferSizeConstants for runtime-sized SSBO arrays.
     * The AIR backend emits code that reads uint32 byte-sizes from a
     * constant uint* buffer at MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX when a
     * shader uses .length() on unsized SSBO arrays.  The pure fill (slot
     * cap / self-slot exclusion / uint32 truncation) lives in the C++
     * facade mglRenderBuildRuntimeArraySizes; this side only extracts the
     * per-buffer {slot, visible-size} pairs from the GL buffer map. */
    {
        Program *computeProgram = mglResolveProgramForStageFromState(ctx, stage);
        if (computeProgram &&
            computeProgram->modules[stage].needs_runtime_array_size_buffer)
        {
            const GLuint runtimeSizeSlot =
                mglRuntimeArraySizeBufferIndexForProgram(computeProgram, stage);
            uint32_t sizeConstants[kMGLMaxMetalVertexBufferCount];
            memset(sizeConstants, 0, sizeof(sizeConstants));

            MGLRenderBufferSizeEntry entries[32]; /* MAX_MAPPED_BUFFERS */
            uint32_t entryCount = 0;
            for (int i = 0; i < bufferMap->count && entryCount < 32; i++)
            {
                BufferMap *map = &bufferMap->buffers[i];
                if (!map->buf) {
                    continue;
                }
                size_t metalSlot = map->has_metal_binding
                    ? (size_t)map->metal_binding_index
                    : (size_t)map->buffer_base_index;
                GLsizeiptr visibleSize = mglBufferMapVisibleSize(map);
                entries[entryCount].metal_slot = (uint32_t)metalSlot;
                entries[entryCount].visible_size = (uint64_t)visibleSize;
                entryCount++;
            }

            if (mglRenderBuildRuntimeArraySizes(
                    entries, entryCount,
                    runtimeSizeSlot,
                    kMGLMaxMetalVertexBufferCount,
                    sizeConstants, kMGLMaxMetalVertexBufferCount) != 0) {
                fprintf(stderr,
                        "MGL COMPUTE ERROR: runtime-array-size constants build failed\n");
                mglComputeBindFlushSnapshot(&bind);
                return false;
            }

            void *sizeBuffer = mglComputeBindCreateBufferWithBytes(
                sizeConstants, sizeof(sizeConstants), 0u);
            if (sizeBuffer) {
                mglComputeBindEmitBuffer(&bind, runtimeSizeSlot, sizeBuffer, 0);
                mglComputeBindHandOffCreated(&bind, sizeBuffer);
                /* sizeBuffer is a block-local (__bridge_transfer on gate-on):
                 * flush before the block ends so the encoder retains it. */
                mglComputeBindFlushSnapshot(&bind);
            }
        }
    }

    mglComputeBindFlushSnapshot(&bind);
    return bind.snapshot_ok;
}

/* === texture / sampler half =============================================
 * The MGL_CTEX_* macros of the Objective-C file, over the same context; kind 2
 * is a texture binding and kind 3 a sampler. */

static void mglComputeBindEmitTexture(MGLComputeBindCtx *ctx, uint32_t slot,
                                      void *texture)
{
    if (ctx->snapshot.op_count >= MGL_RENDER_COMPUTE_BINDING_SNAPSHOT_MAX_OPS) {
        mglComputeBindFlushSnapshot(ctx);
    }
    ctx->snapshot.ops[ctx->snapshot.op_count++] =
        (MGLRenderComputeBindingOp){/* kind */ 2u,
                                    /* index */ slot,
                                    /* offset */ 0,
                                    /* buffer */ texture,
                                    /* bytes */ NULL,
                                    /* length */ 0u};
}

static void mglComputeBindEmitSampler(MGLComputeBindCtx *ctx, uint32_t slot,
                                      void *sampler)
{
    if (ctx->snapshot.op_count >= MGL_RENDER_COMPUTE_BINDING_SNAPSHOT_MAX_OPS) {
        mglComputeBindFlushSnapshot(ctx);
    }
    ctx->snapshot.ops[ctx->snapshot.op_count++] =
        (MGLRenderComputeBindingOp){/* kind */ 3u,
                                    /* index */ slot,
                                    /* offset */ 0,
                                    /* buffer */ sampler,
                                    /* bytes */ NULL,
                                    /* length */ 0u};
}

bool mglComputeBindTexturesToEncoder(void *renderer, int stage, void *encoder,
                                     MGLRenderComputeExecutionPlan *plan,
                                     void *temporaries)
{
    if (!encoder && !plan) {
        fprintf(stderr,
                "MGL COMPUTE ERROR: NULL compute encoder for texture binding\n");
        return false;
    }

    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    GLMState *state = mglComputeBindState(&areas);

    MGLComputeBindCtx bind = {
        .encoder = encoder,
        .plan = plan,
        .temporaries = temporaries,
        .snapshot_ok = true,
    };
    /* The Objective-C macro created a local NSMutableArray when none was
     * passed and the function dropped it again on return; that set therefore
     * never reached a caller.  Created and released the same way here. */
    bool owns_temporaries = false;

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

        /* iterate shader storage buffers */
        GLuint count = mglRendererGetProgramBindingCount(ctx, stage, spvc_type);
        if (count)
        {
            int textures_to_be_mapped = count;

            if (textures_to_be_mapped > TEXTURE_UNITS) {
                textures_to_be_mapped = TEXTURE_UNITS;
            }

            for (int i = 0; i < (int)count && textures_to_be_mapped > 0; i++)
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

                if (mglRenderComputeTextureBindIsStorage((uint32_t)gl_texture_type))
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
                                      ->sampler_units_by_stage[stage][metalBinding]
                                : 0u,
                            resource ? resource->sampler_unit : -1,
                            resource ? resource->gl_binding : 0u,
                            resourceElement);
                    } else {
                        glUnit = (GLuint)mglRendererGetProgramGLBinding(
                            ctx, stage, spvc_type, i);
                    }
                    if (!mglRenderImageUnitsInRange(0u, glUnit, TEXTURE_UNITS)) {
                        continue;
                    }
                    ptr = state->image_units[glUnit].tex;
                } else {
                    glUnit = mglTextureUnitForSampledResource(
                        resource, mglResolveProgramForStageFromState(ctx, stage),
                        metalBinding, stage);
                    if (glUnit >= TEXTURE_UNITS) {
                        continue;
                    }
                    ptr = mglTextureForSampledResourceForStage(
                        ctx, resource, metalBinding, stage,
                        mglRendererGetProgramDeclaredTextureType(ctx, stage,
                                                                 spvc_type, i));
                }

                if (ptr)
                {
                    RETURN_FALSE_ON_FAILURE(mglRendererBindMTLTexture(renderer, ptr));
                    if (!ptr->mtl_data) {
                        continue;
                    }

                    void *texture = ptr->mtl_data;
                    if (!texture) {
                        continue;
                    }

                    /* Storage images: BindImage <format>/level/slice views
                     * (same helper as VS/FS). Cached on ImageUnit. */
                    if (mglRenderComputeTextureBindIsStorage(
                            (uint32_t)gl_texture_type)) {
                        texture = mglRendererStorageImageTexture(
                            texture, &state->image_units[glUnit]);
                    }

                    /* Sampler cascade (GL sampler object → texture parameters
                     * → default) is the shared materialize port the
                     * vertex / fragment spine uses. */
                    void *sampler = mglRendererMaterializeSampledSamplerPort(
                        renderer, ptr, glUnit, NULL, 0, (uint32_t)ptr->target,
                        computeProgram ? computeProgram->name : 0u,
                        resource ? mglMetalResourceSlot(resource) : metalBinding,
                        "compute", texture);

                    if (!sampler) {
                        void *fallbackSampler = NULL;
                        if (mglRenderCreateDefaultSampler(&fallbackSampler) != 0) {
                            fallbackSampler = NULL;
                        }
                        sampler = fallbackSampler;
                        /* Keep the fallback alive until the end replay. */
                        if (!temporaries) {
                            temporaries = mglRendererTemporariesCreate();
                            bind.temporaries = temporaries;
                            owns_temporaries = true;
                        }
                        mglComputeBindHandOffCreated(&bind, fallbackSampler);
                        if (!sampler) {
                            continue;
                        }
                    }

                    mglComputeBindEmitTexture(&bind, metalBinding, texture);
                    if (mglRenderComputeTextureBindNeedsSampler(
                            (uint32_t)gl_texture_type,
                            !resource || resource->has_combined_sampler)) {
                        GLuint samplerBinding = resource
                            ? mglMetalCombinedSamplerSlotForElement(resource,
                                                                    resourceElement)
                            : metalBinding;
                        mglComputeBindEmitSampler(&bind, samplerBinding, sampler);
                    }

                    textures_to_be_mapped--;
                }
            }

            /* texture not found */
            if (textures_to_be_mapped)
            {
                DEBUG_PRINT("No texture bound for fragment shader location\n");
                mglComputeBindFlushSnapshot(&bind);
                if (owns_temporaries) {
                    mglRendererTemporariesRelease(temporaries);
                }
                return false;
            }
        }
    }

    if (computeProgram) {
        MGLShaderResourceList *arrayResources =
            &computeProgram->shader_resources_list[stage][_SAMPLED_IMAGE_RES];
        for (GLuint resourceIndex = 0;
             arrayResources->list && resourceIndex < arrayResources->count;
             resourceIndex++) {
            MGLShaderResource *resource = &arrayResources->list[resourceIndex];
            if (resource->gl_array_size <= 1) {
                continue;
            }

            uint32_t expectedType = mglRendererGetProgramDeclaredTextureType(
                ctx, stage, _SAMPLED_IMAGE_RES, (int)resourceIndex);
            for (GLint element = 1; element < resource->gl_array_size; element++) {
                GLuint metalSlot = resource->binding + (GLuint)element;
                GLuint samplerSlot = mglMetalCombinedSamplerSlotForElement(
                    resource, (GLuint)element);
                if (metalSlot >= TEXTURE_UNITS) {
                    break;
                }

                GLuint glUnit = mglTextureUnitForSampledResource(
                    NULL, mglResolveProgramForStageFromState(ctx, stage),
                    metalSlot, stage);
                Texture *ptr = mglTextureForSampledResourceForStage(
                    ctx, NULL, metalSlot, stage, expectedType);
                if (!ptr || !mglRendererBindMTLTexture(renderer, ptr) ||
                    !ptr->mtl_data) {
                    continue;
                }

                void *texture = ptr->mtl_data;
                /* Same shared port as the loop above.  This path used to
                 * skip the "dirty sampler" release the others do, so a
                 * re-parameterized sampler could keep its old Metal object;
                 * going through the port makes it consistent. */
                void *sampler = mglRendererMaterializeSampledSamplerPort(
                    renderer, ptr, glUnit, NULL, 0, (uint32_t)ptr->target,
                    computeProgram ? computeProgram->name : 0u, metalSlot,
                    "compute", texture);

                if (!sampler) {
                    void *fallbackSampler = NULL;
                    if (mglRenderCreateDefaultSampler(&fallbackSampler) != 0) {
                        fallbackSampler = NULL;
                    }
                    sampler = fallbackSampler;
                    /* Keep the fallback alive until the end replay. */
                    if (!temporaries) {
                        temporaries = mglRendererTemporariesCreate();
                        bind.temporaries = temporaries;
                        owns_temporaries = true;
                    }
                    mglComputeBindHandOffCreated(&bind, fallbackSampler);
                }

                mglComputeBindEmitTexture(&bind, metalSlot, texture);
                if (resource->has_combined_sampler && sampler) {
                    mglComputeBindEmitSampler(&bind, samplerSlot, sampler);
                }
            }
        }
    }

    mglComputeBindFlushSnapshot(&bind);
    if (owns_temporaries) {
        mglRendererTemporariesRelease(temporaries);
    }

    state->dirty_bits &= ~(DIRTY_TEX_BINDING | DIRTY_SAMPLER | DIRTY_IMAGE_UNIT_STATE);

    return bind.snapshot_ok;
}
