/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_tess_stage_bind.c — the tessellation stage-binding plan and the texture
 * binding plan moved out of MGLRenderer+Tessellation.m (P0-1, log 125).
 *
 * Mechanical translation of five methods:
 *
 *   -prepareTessStageBufferBindings:stage:copyBacks:   -> mglTessPrepareStageBufferBindings
 *   -flushTessStageBindingInitializationBlit:          -> mglTessFlushStageBindingInitializationBlit
 *   -bindTessStageBufferBindingsToRenderEncoderOwner:  -> mglTessBindStageBufferBindingsToRenderEncoderOwner
 *   -bindPreparedTessStageBufferBindings:...           -> mglTessBindPreparedStageBufferBindings
 *   -planTessTextureBinds:count:ctx:plan:temporaries:  -> mglTessPlanTextureBinds
 *
 * The rules the translation follows are the ones the earlier cuts settled:
 * `self` becomes the renderer handle, `_renderPassManager->state` and the
 * tessellation record arrive through MGLRendererStateAreas, `ctx` is
 * `areas.ctx`, `MGL_STATE(ctx)` becomes the C twin below, "[self ...]" becomes
 * the C entry of mgl_renderer_ports.h, and the NSMutableArray of temporaries
 * becomes the handle of mglRendererTemporariesCreate (which is exactly the
 * array: the port retains it and the C caller drops its creation reference).
 */

#include <stdio.h>
#include <string.h>

#include <CoreFoundation/CoreFoundation.h>

#include "mgl_tess_stage_bind.h"
#include "mgl_renderer_ports.h"     /* state areas + stage-binding host entries */
#include "mgl_renderer_backend.h"   /* program binding sizes, storage-image texture */
#include "mgl_render.h"             /* buffer/texture creation, copy encoding */
#include "mgl_buffer_map.h"         /* map entries, dirty update, mglNoteBufferEncoded */
#include "mgl_buffer_slots.h"       /* mglRuntimeArraySizeBufferIndexForProgram */
#include "mgl_binding_stage.h"      /* bind plan for a map entry */
#include "mgl_texture_sampler.h"    /* mglTextureCreateSamplerForTexParam */
#include "mgl_texture_bind.h"       /* mglRendererBindMTLBuffer */
#include "mgl_metal_ref.h"          /* mglSafeReleaseMetalObj */
#include "mgl_thread_affinity.h"    /* MGL_ASSERT_GL_THREAD */
#include "mgl_types_texture.h"      /* Texture, Sampler */

/* Declared next to its definition in the Objective-C
 * MGLRenderer+RenderPass_Private.h, which a .c file cannot include; repeated
 * here the way mgl_renderer_ports.c repeats the prototypes it needs. */
extern Program *mglResolveProgramForStageFromState(GLMContext ctx, int stage);

/* MGL_STATE() from MGLRenderer_Private.h, in C (the same twin as
 * mgl_compute_bind.c). */
static GLMState *mglTessStageBindState(const MGLRendererStateAreas *areas)
{
    if (areas->core && areas->core->activeState) {
        return areas->core->activeState;
    }
    return areas->ctx ? areas->ctx->active_state : NULL;
}

/* The command record the render pass manager publishes. */
static void *mglTessStageBindCommandBufferOwner(
    const MGLRendererStateAreas *areas)
{
    return areas->command ? areas->command->currentCommandBufferOwner : NULL;
}

static void *mglTessStageBindRenderEncoderOwner(
    const MGLRendererStateAreas *areas)
{
    return areas->command ? areas->command->currentRenderEncoderOwner : NULL;
}

/* === File-local twins of the Objective-C statics ==========================
 * MGLRenderer+Tessellation.m still owns the `id`-typed statics for the methods
 * that have not moved yet; these are the C twins over the same mglRender*
 * entries, the pattern mgl_draw_metal_port.c uses for mglVboRangeValidationEnabled. */

/* The .m's MGL_TESS_RESOURCE_STORAGE_SHARED: MTLResourceStorageModeShared. */
#define MGL_TESS_STAGE_BIND_STORAGE_SHARED 0u

/* +1 buffer, or NULL. */
static void *mglTessStageBindCreateBuffer(size_t length, uint64_t options)
{
    void *buffer = NULL;
    if (mglRenderCreateBuffer((uint64_t)length, options, NULL, &buffer) == 0 &&
        buffer) {
        return buffer;
    }
    return NULL;
}

/* +1 buffer, or NULL. */
static void *mglTessStageBindCreateBufferWithBytes(const void *bytes,
                                                   size_t length,
                                                   uint64_t options)
{
    void *buffer = NULL;
    if (mglRenderCreateBufferWithBytes(bytes, length, options, NULL, &buffer) ==
            0 &&
        buffer) {
        return buffer;
    }
    return NULL;
}

/* +1 default sampler, or NULL. */
static void *mglTessStageBindCreateSampler(void)
{
    void *sampler = NULL;
    if (mglRenderCreateDefaultSampler(&sampler) == 0 && sampler) {
        return sampler;
    }
    return NULL;
}

static uint64_t mglTessStageBindBufferLength(void *buffer)
{
    MGLRenderBufferInfo info = {0};
    return buffer && mglRenderGetBufferInfo(buffer, &info) == 0 ? info.length
                                                                : 0u;
}

static void *mglTessStageBindBufferContents(void *buffer)
{
    void *contents = NULL;
    uint64_t length = 0u;
    return buffer && mglRenderGetBufferContents(buffer, &contents, &length) == 0
               ? contents
               : NULL;
}

static void mglTessStageBindSetRenderVertexBuffer(void *render_encoder_owner,
                                                  void *buffer,
                                                  size_t offset,
                                                  size_t index)
{
    (void)mglRenderSetRenderBufferForOwner(
        render_encoder_owner, buffer, offset, MGL_RENDER_BINDING_STAGE_VERTEX,
        (uint32_t)index);
}

static bool mglTessStageBindEncodeBufferCopiesForOwner(
    void *command_buffer_owner, const MGLRenderBufferCopyEntry *entries,
    uint32_t entry_count)
{
    if (!command_buffer_owner || !entries || entry_count == 0u) {
        return false;
    }
    return mglRenderEncodeBufferCopiesForCommandBufferOwner(
               command_buffer_owner, entries, entry_count) == 0;
}

static bool mglTessStageBindAppendComputeResourceOp(
    MGLRenderComputeExecutionPlan *plan, void *temporaries, uint32_t kind,
    void *resource, size_t offset, size_t index)
{
    if (!plan || kind > 3u) {
        return false;
    }
    if (plan->binding_op_count >= MGL_RENDER_COMPUTE_EXECUTION_MAX_OPS) {
        fprintf(stderr, "MGL TESS ERROR: compute binding op overflow (%u)",
                (unsigned)plan->binding_op_count);
        return false;
    }
    plan->binding_ops[plan->binding_op_count++] = (MGLRenderComputeBindingOp){
        .kind = kind,
        .index = (uint32_t)index,
        .offset = (uint64_t)offset,
        .buffer = resource,
        .bytes = NULL,
        .length = 0u,
    };
    if (resource && temporaries) {
        mglRendererTemporariesAdd(temporaries, resource);
    }
    return true;
}

static bool mglTessStageBindPlanTextureOrBind(
    MGLRenderComputeExecutionPlan *plan, void *temporaries, void *texture,
    size_t index)
{
    return mglTessStageBindAppendComputeResourceOp(plan, temporaries, 2u,
                                                   texture, 0u, index);
}

static bool mglTessStageBindPlanSamplerOrBind(
    MGLRenderComputeExecutionPlan *plan, void *temporaries, void *sampler,
    size_t index)
{
    return mglTessStageBindAppendComputeResourceOp(plan, temporaries, 3u,
                                                   sampler, 0u, index);
}

/* === The moved methods ==================================================== */

/* Tessellation shaders run as consecutive compute encoders. Prepare their
 * buffer bindings before opening the next encoder so an isolated binding can
 * be initialized by an ordered GPU copy from a buffer written by the previous
 * stage. Reading source.contents here would capture stale CPU bytes while the
 * preceding TCS encoder is still pending on the same command buffer. */
bool mglTessPrepareStageBufferBindings(void *renderer,
                                       MGLTessStageBufferBindingList *bindings,
                                       int stage,
                                       MGLStageBindingCopyBackList *copy_backs)
{
    MGL_ASSERT_GL_THREAD();
    if (!renderer || !bindings || !copy_backs) {
        return false;
    }

    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    BufferMapList stage_buffer_map = {0};
    if (!mglRendererMapGLBuffersToMTLBufferMap(renderer, &stage_buffer_map,
                                               stage)) {
        return false;
    }

    /* Complete every lazy allocation before creating the initialization blit
     * encoder. bindMTLBuffer: may itself need an encoder. */
    for (GLuint i = 0; i < stage_buffer_map.count; i++) {
        Buffer *ptr = stage_buffer_map.buffers[i].buf;
        if (ptr && !ptr->data.mtl_data) {
            mglRendererBindMTLBuffer(renderer, ptr);
        }
    }

    for (GLuint i = 0; i < stage_buffer_map.count; i++) {
        BufferMap *map = &stage_buffer_map.buffers[i];
        Buffer *ptr = map->buf;
        if (!ptr) {
            continue;
        }

        uint32_t metal_binding_index = 0u;
        if (!mglRenderResolveMappedBufferSlot(
                map->has_metal_binding ? 1 : 0,
                (int32_t)map->metal_binding_index, (int32_t)map->buffer_base_index,
                (uint32_t)kMGLMaxMetalVertexBufferCount, &metal_binding_index)) {
            continue;
        }
        mglRendererClearStageBindingCopyBackPort(renderer, copy_backs,
                                                 metal_binding_index);
        void *buffer = ptr->data.mtl_data;
        if (buffer && mglRenderBufferHasCPUDirty(ptr->data.dirty_bits)) {
            /* Consume the CPU-side initialization before a tessellation
             * stage can write the same Metal backing. Otherwise a later
             * stage bind would upload the stale shadow over the GPU result. */
            if (!mglRendererUpdateDirtyBuffer(renderer, ptr)) {
                return false;
            }
            buffer = ptr->data.mtl_data;
        }
        MGLTessIsolatedBindingPlan bind_plan = {0};
        GLsizeiptr storage_remaining = mglBufferMapStorageRemaining(map);
        const uint64_t buffer_length = mglTessStageBindBufferLength(buffer);
        size_t available_bytes =
            buffer ? mglBufferMapVisibleBackingBytes(map, buffer_length) : 0u;
        size_t required_bytes = mglRendererGetProgramBindingRequiredSize(
            areas.ctx, stage, (int)map->resource_type, (int)map->resource_index);
        required_bytes = mglTessRequiredBindingBytes((int)map->resource_type,
                                                     (uint32_t)required_bytes);
        if (!mglTessPlanIsolatedBinding(buffer ? 1 : 0, map->offset,
                                        buffer_length,
                                        (int64_t)storage_remaining,
                                        (uint64_t)available_bytes,
                                        (uint32_t)required_bytes,
                                        (int)map->resource_type, &bind_plan)) {
            return false;
        }

        /* A TES-vertex stage reads its SSBO/UBO resources inside the render
         * encoder, so it never writes them and never needs a copy-back.
         * Follows the per-draw path: a program that also carries the compute
         * kernel (indexed draws) must keep isolated bindings on the compute
         * path. */
        const int stage_reads_only =
            (stage == _TESS_EVALUATION_SHADER &&
             areas.tessellation->tessVertexRenderActive)
                ? 1
                : 0;
        if (stage_reads_only) {
            bind_plan.isolated = 0;
        }

        MGLTessStageBufferBinding *binding = &bindings->slots[metal_binding_index];
        binding->buffer = NULL;
        binding->offset = 0u;
        binding->initialization_source = NULL;
        binding->initialization_source_offset = 0u;
        binding->initialization_length = 0u;
        binding->valid = 1;
        if (!bind_plan.isolated) {
            binding->buffer = buffer;
            binding->offset = (size_t)map->offset;
            /* The GL buffer's Metal backing is about to be staged in a
             * compute encoder: pin its snapshot-pool slot. */
            mglNoteBufferEncoded(ptr);
            continue;
        }

        size_t fallback_length = bind_plan.fallback_length;
        void *isolated = mglTessStageBindCreateBuffer(
            fallback_length, MGL_TESS_STAGE_BIND_STORAGE_SHARED);
        void *isolated_contents = mglTessStageBindBufferContents(isolated);
        if (!isolated_contents) {
            return false;
        }
        memset(isolated_contents, 0, fallback_length);

        binding->buffer = isolated;
        binding->offset = 0u;
        if (bind_plan.init_length > 0u) {
            binding->initialization_source = buffer;
            binding->initialization_source_offset = (size_t)map->offset;
            binding->initialization_length = bind_plan.init_length;
        }

        if (mglTessIsolatedNeedsCopyBack(bind_plan.writable ? 1 : 0,
                                         buffer ? 1 : 0,
                                         bind_plan.init_length) &&
            !mglRendererRecordStageBindingCopyBackPort(
                renderer, copy_backs, metal_binding_index, isolated, buffer, ptr,
                (size_t)map->offset, available_bytes)) {
            return false;
        }
    }

    Program *stage_program = mglResolveProgramForStageFromState(areas.ctx, stage);
    if (stage_program &&
        stage_program->modules[stage].needs_runtime_array_size_buffer) {
        bindings->size_buffer_index =
            mglRuntimeArraySizeBufferIndexForProgram(stage_program, stage);
        uint32_t size_constants[kMGLMaxBufferSlots] = {0};
        mglTessFillRuntimeArraySizeConstants(
            stage_buffer_map.buffers, stage_buffer_map.count,
            bindings->size_buffer_index, size_constants, kMGLMaxBufferSlots);
        bindings->size_buffer = mglTessStageBindCreateBufferWithBytes(
            size_constants, sizeof(size_constants),
            MGL_TESS_STAGE_BIND_STORAGE_SHARED);
        if (!bindings->size_buffer) {
            return false;
        }
    }

    /* A TES-vertex stage binds its resources into the render encoder; the
     * GPU copy that initializes an isolated binding must land before that
     * encoder (it is encoded into the command buffer, not the render pass). */
    const int tess_vertex_render_stage =
        (stage == _TESS_EVALUATION_SHADER &&
         areas.tessellation->tessVertexRenderActive)
            ? 1
            : 0;
    if (tess_vertex_render_stage) {
        int needs_initialization_blit = 0;
        for (size_t i = 0; i < kMGLMaxBufferSlots; i++) {
            if (bindings->slots[i].initialization_length > 0) {
                needs_initialization_blit = 1;
                break;
            }
        }
        if (needs_initialization_blit) {
            if (!mglTessFlushStageBindingInitializationBlit(renderer, bindings)) {
                return false;
            }
        }
    }

    int needs_initialization_blit = 0;
    for (size_t i = 0; i < kMGLMaxBufferSlots; i++) {
        if (bindings->slots[i].initialization_length > 0) {
            needs_initialization_blit = 1;
            break;
        }
    }
    if (!needs_initialization_blit) {
        return true;
    }
    MGLRenderCommandBufferState command_state = {0};
    const int has_command_state = mglRenderCommandBufferOwnerHasState(
        mglTessStageBindCommandBufferOwner(&areas), &command_state);
    if (!mglTessCommandBufferCanInitBlit(has_command_state,
                                         command_state.status)) {
        return false;
    }

    MGLRenderBufferCopyEntry copy_entries[kMGLMaxBufferSlots] = {0};
    uint32_t copy_entry_count = 0u;
    for (size_t i = 0; i < kMGLMaxBufferSlots; i++) {
        MGLTessStageBufferBinding *binding = &bindings->slots[i];
        if (binding->initialization_length == 0) {
            continue;
        }
        copy_entries[copy_entry_count++] = (MGLRenderBufferCopyEntry){
            .source_buffer = binding->initialization_source,
            .source_offset = binding->initialization_source_offset,
            .destination_buffer = binding->buffer,
            .destination_offset = 0u,
            .length = binding->initialization_length,
        };
    }
    return mglTessStageBindEncodeBufferCopiesForOwner(
        mglTessStageBindCommandBufferOwner(&areas), copy_entries,
        copy_entry_count);
}

/* Encode the isolated-binding initialization copies into the command buffer
 * now (used by the TES-vertex path, which binds resources into a render
 * encoder rather than a compute plan). */
bool mglTessFlushStageBindingInitializationBlit(
    void *renderer, MGLTessStageBufferBindingList *bindings)
{
    MGL_ASSERT_GL_THREAD();
    if (!renderer || !bindings) {
        return false;
    }

    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    MGLRenderBufferCopyEntry copy_entries[kMGLMaxBufferSlots] = {0};
    uint32_t copy_entry_count = 0u;
    for (size_t i = 0; i < kMGLMaxBufferSlots; i++) {
        MGLTessStageBufferBinding *binding = &bindings->slots[i];
        if (binding->initialization_length == 0) {
            continue;
        }
        copy_entries[copy_entry_count++] = (MGLRenderBufferCopyEntry){
            .source_buffer = binding->initialization_source,
            .source_offset = binding->initialization_source_offset,
            .destination_buffer = binding->buffer,
            .destination_offset = 0u,
            .length = binding->initialization_length,
        };
    }
    if (copy_entry_count == 0u) {
        return true;
    }
    if (mglTessMustEndRenderBeforeCompute(
            mglRenderEncoderOwnerHasCurrent(
                mglTessStageBindRenderEncoderOwner(&areas)))) {
        mglRendererEndRenderEncodingPort(renderer);
    }
    return mglTessStageBindEncodeBufferCopiesForOwner(
        mglTessStageBindCommandBufferOwner(&areas), copy_entries,
        copy_entry_count);
}

/* Bind the prepared TES stage buffers (SSBO/UBO/atomic/runtime-size) into the
 * render encoder for the TES-vertex path.  These are read-only in the vertex
 * stage; no copy-back is recorded. */
bool mglTessBindStageBufferBindingsToRenderEncoderOwner(
    void *render_encoder_owner, const MGLTessStageBufferBindingList *bindings)
{
    MGL_ASSERT_GL_THREAD();
    if (!bindings) {
        return false;
    }
    for (size_t i = 0; i < kMGLMaxBufferSlots; i++) {
        const MGLTessStageBufferBinding *binding = &bindings->slots[i];
        if (binding->valid && binding->buffer) {
            mglTessStageBindSetRenderVertexBuffer(render_encoder_owner,
                                                  binding->buffer,
                                                  binding->offset, i);
        }
    }
    if (bindings->size_buffer) {
        mglTessStageBindSetRenderVertexBuffer(
            render_encoder_owner, bindings->size_buffer, 0u,
            bindings->size_buffer_index);
    }
    return true;
}

bool mglTessBindPreparedStageBufferBindings(
    const MGLTessStageBufferBindingList *bindings, void *compute_command_encoder,
    MGLRenderComputeExecutionPlan *execution_plan, void *temporaries)
{
    MGL_ASSERT_GL_THREAD();
    (void)compute_command_encoder;
    if (!bindings || !execution_plan) {
        return false;
    }
    for (size_t i = 0; i < kMGLMaxBufferSlots; i++) {
        const MGLTessStageBufferBinding *binding = &bindings->slots[i];
        if (binding->valid) {
            if (!mglTessStageBindAppendComputeResourceOp(
                    execution_plan, temporaries, 0u, binding->buffer,
                    binding->offset, i)) {
                return false;
            }
        }
    }
    if (bindings->size_buffer) {
        if (!mglTessStageBindAppendComputeResourceOp(
                execution_plan, temporaries, 0u, bindings->size_buffer, 0u,
                bindings->size_buffer_index)) {
            return false;
        }
    }
    return true;
}

bool mglTessPlanTextureBinds(void *renderer, const MGLTessTextureBind *binds,
                             uint32_t count, GLMContext ctx,
                             MGLRenderComputeExecutionPlan *plan,
                             void *temporaries)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    if (!binds || !plan || !ctx || !ctx->active_state) {
        return binds == NULL || count == 0u;
    }
    for (uint32_t i = 0; i < count; i++) {
        const MGLTessTextureBind *bind = &binds[i];
        void *texture = NULL;
        Texture *ptr = NULL;
        GLMState *state = mglTessStageBindState(&areas);
        if (mglTessTextureBindIsStorage(bind->kind)) {
            ptr = state->image_units[bind->gl_unit].tex;
            if (ptr) {
                texture = ptr->mtl_data;
                texture = mglRendererStorageImageTexture(
                    texture, &state->image_units[bind->gl_unit]);
            }
        } else {
            ptr = state->active_textures[bind->gl_unit];
            texture = ptr ? ptr->mtl_data : NULL;
        }
        if (!mglTessStageBindPlanTextureOrBind(plan, temporaries, texture,
                                               bind->metal_slot)) {
            return false;
        }
        if (!mglTessTextureBindNeedsSampler(bind->kind,
                                            bind->combined_sampler_slot)) {
            continue;
        }
        void *sampler = NULL;
        int created_sampler = 0;
        if (state->texture_samplers[bind->gl_unit]) {
            Sampler *gl_sampler = state->texture_samplers[bind->gl_unit];
            if (gl_sampler->dirty_bits && gl_sampler->mtl_data) {
                mglSafeReleaseMetalObj((void **)&gl_sampler->mtl_data);
            }
            if (!gl_sampler->mtl_data && ptr) {
                /* The C creation hands back +1; the field owns it. */
                gl_sampler->mtl_data = mglTextureCreateSamplerForTexParam(
                    &gl_sampler->params, ptr->target);
                gl_sampler->dirty_bits = 0;
            }
            sampler = gl_sampler->mtl_data;
        } else if (ptr && ptr->params.mtl_data) {
            sampler = ptr->params.mtl_data;
        }
        if (!sampler) {
            sampler = mglTessStageBindCreateSampler();
            created_sampler = 1;
        }
        if (sampler) {
            const bool planned = mglTessStageBindPlanSamplerOrBind(
                plan, temporaries, sampler, bind->combined_sampler_slot);
            if (created_sampler) {
                /* The temporaries set kept its own reference; a failure still
                 * has to drop the creation reference. */
                CFRelease((CFTypeRef)sampler);
            }
            if (!planned) {
                return false;
            }
        }
    }
    return true;
}
