/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_storage_image_bind.c — the storage-image binding driver moved out of
 * MGLRenderer+BindingState.m (P0-1, log 130).
 *
 * Mechanical translation: `self` becomes the renderer handle, `ctx` is
 * `areas.ctx`, `_bindingStateOwner` is `*areas.binding_state_owner`,
 * `_renderPassManager->state->currentRenderEncoderOwner` is the command
 * record's encoder owner, and the Objective-C header's static inline helpers
 * (`mglBindingStateQueueResourceBinding` / …FlushResourceBindings /
 * …CollectResourceBinding) plus the file statics
 * (`mglBindingStateResourceAtOrdinal` / …CreateStorageImageView) are repeated
 * here as the C twins `mglSi*`, the pattern the earlier cuts established.
 * The method-local MGL_ABORT_TBIND_IF_ENCODER_CLOSED macro becomes
 * mglSiAbortTBindIfEncoderClosed(), which the loop calls explicitly.
 *
 * OWNERSHIP (log 128 rule): this driver keeps no `id` local alive beyond the
 * call — `texture` is a borrowed view the snapshot records, and the encoder
 * retains it when the snapshot is flushed inside this function.
 */

#include <stdbool.h>
#include <stdint.h>

#include "mgl_storage_image_bind.h"
#include "mgl_renderer_ports.h"   /* state areas, texture bind, restore encoder */
#include "mgl_renderer_backend.h" /* program binding counts/GL bindings */
#include "mgl_render.h"           /* resource binding snapshot + owners */
#include "mgl_binding_texture.h"  /* storage-image plan */
#include "mgl_binding_policy.h"   /* mglRenderResourceMetalSlot */
#include "mgl_texture_bind.h"     /* mglRendererBindMTLTexture */
#include "mgl_types_state.h"      /* mglMarkRendererDirtyBits + DIRTY_* */
#include "mgl_metal_ref.h"        /* shared metal reference helpers */
#include "glm_context.h"          /* RETURN_FALSE_ON_FAILURE */

/* MGL_STATE() from MGLRenderer_Private.h, in C (the same twin as
 * mgl_compute_bind.c). */
static GLMState *mglSiState(const MGLRendererStateAreas *areas)
{
    if (areas->core && areas->core->activeState) {
        return areas->core->activeState;
    }
    return areas->ctx ? areas->ctx->active_state : NULL;
}

/* The file statics of MGLRenderer+BindingState.m, in C. */
static void *mglSiCreateStorageImageView(void *texture, ImageUnit *iu)
{
    return mglRendererStorageImageTexture(texture, iu);
}

static MGLShaderResource *mglSiResourceAtOrdinal(Program *program, int stage,
                                                 int res_type, GLuint ordinal,
                                                 GLuint *element_out)
{
    if (element_out) {
        *element_out = 0u;
    }
    if (!program || stage < 0 || res_type < 0) {
        return NULL;
    }
    MGLShaderResourceList *list = &program->shader_resources_list[stage][res_type];
    GLuint rem = ordinal;
    for (GLuint ri = 0; ri < list->count; ri++) {
        GLuint elements =
            mglRenderShaderResourceElementCount((uint32_t)list->list[ri].gl_array_size);
        if (rem < elements) {
            if (element_out) {
                *element_out = rem;
            }
            return &list->list[ri];
        }
        rem -= elements;
    }
    return NULL;
}

/* The Objective-C header's MGLRenderer resource-binding helpers, in C. */
static bool mglSiCollectResourceBinding(MGLRenderResourceBindingSnapshot *snapshot,
                                        uint32_t stage, uint32_t kind,
                                        void *resource, uint32_t index)
{
    if (!snapshot || stage > MGL_RENDER_BINDING_STAGE_FRAGMENT ||
        kind > MGL_RENDER_RESOURCE_BINDING_SAMPLER) {
        return false;
    }
    uint32_t *count = stage == MGL_RENDER_BINDING_STAGE_VERTEX
                          ? &snapshot->vertex_op_count
                          : &snapshot->fragment_op_count;
    MGLRenderResourceBindingOp *ops = stage == MGL_RENDER_BINDING_STAGE_VERTEX
                                          ? snapshot->vertex_ops
                                          : snapshot->fragment_ops;
    if (*count >= MGL_RENDER_RESOURCE_BINDING_SNAPSHOT_MAX_OPS) {
        return false;
    }
    ops[(*count)++] = (MGLRenderResourceBindingOp){
        .kind = kind,
        .index = index,
        .resource = resource,
    };
    return true;
}

static bool mglSiQueueResourceBinding(int collect, void *binding_state_owner,
                                      void *render_encoder_owner,
                                      MGLRenderResourceBindingSnapshot *snapshot,
                                      uint32_t stage, uint32_t kind,
                                      void *resource, uint32_t index)
{
    if (collect) {
        return mglSiCollectResourceBinding(snapshot, stage, kind, resource, index);
    }
    if (kind == MGL_RENDER_RESOURCE_BINDING_TEXTURE) {
        return mglRenderBindingSetTextureForOwner(binding_state_owner,
                                                  render_encoder_owner, resource,
                                                  stage, index) >= 0;
    }
    if (kind == MGL_RENDER_RESOURCE_BINDING_SAMPLER) {
        return mglRenderBindingSetSamplerForOwner(binding_state_owner,
                                                  render_encoder_owner, resource,
                                                  stage, index) >= 0;
    }
    return false;
}

static bool mglSiFlushResourceBindings(void *binding_state_owner,
                                       void *render_encoder_owner,
                                       MGLRenderResourceBindingSnapshot *snapshot)
{
    if (!snapshot || (snapshot->vertex_op_count == 0 &&
                      snapshot->fragment_op_count == 0)) {
        return true;
    }
    if (mglRenderEncodeResourceBindingSnapshotForRenderEncoderOwner(
            binding_state_owner, render_encoder_owner, snapshot, NULL, 0) != 0) {
        return false;
    }
    *snapshot = (MGLRenderResourceBindingSnapshot){0};
    return true;
}

/* The render encoder owner is re-read at every use, exactly like the method
 * did: a texture upload inside the ENSURE pass restores (and therefore
 * replaces) it, and the queue/flush must target the current one. */
static void *mglSiRenderEncoderOwner(const MGLRendererStateAreas *areas)
{
    return areas->command ? areas->command->currentRenderEncoderOwner : NULL;
}

/* The method-local MGL_ABORT_TBIND_IF_ENCODER_CLOSED(): true when the draw
 * must be abandoned because the texture path closed the render encoder. */
static bool mglSiAbortTBindIfEncoderClosed(const MGLRendererStateAreas *areas)
{
    if (mglRenderEncoderOwnerHasCurrent(mglSiRenderEncoderOwner(areas)) == 0) {
        if (areas->ctx) {
            mglMarkRendererDirtyBits(areas->ctx->active_state,
                                     (DIRTY_TEX | DIRTY_TEX_BINDING |
                                      DIRTY_RENDER_STATE));
        }
        return true;
    }
    return false;
}

bool mglBindingStateBindStorageImagesForStage(void *renderer, int shader_stage,
                                              Program *program,
                                              uint32_t metal_bind_stage)
{
    if (!renderer) {
        return false;
    }

    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    void *binding_owner =
        areas.binding_state_owner ? *areas.binding_state_owner : NULL;
    const int use_resource_snapshot = 1;
    MGLRenderResourceBindingSnapshot resource_snapshot = {0};
    GLuint count =
        mglRendererGetProgramBindingCount(areas.ctx, shader_stage, _STORAGE_IMAGE_RES);
    const char *restore_tag =
        metal_bind_stage == MGL_RENDER_BINDING_STAGE_VERTEX
            ? "vs-storage-image-bind"
            : "storage-image-bind";

    for (int pass = MGL_SI_PASS_ENSURE; pass <= MGL_SI_PASS_BIND; pass++) {
        for (GLuint i = 0; i < count; i++) {
            GLuint element = 0u;
            MGLShaderResource *resource = mglSiResourceAtOrdinal(
                program, shader_stage, _STORAGE_IMAGE_RES, i, &element);
            const uint32_t fallback_metal = (GLuint)mglRendererGetProgramBinding(
                areas.ctx, shader_stage, _STORAGE_IMAGE_RES, (int)i);
            const uint32_t provisional_slot = mglRenderResourceMetalSlot(
                resource ? 1 : 0, resource ? resource->binding : 0u, element,
                fallback_metal);
            const int explicit_unit =
                program && provisional_slot < TEXTURE_UNITS &&
                program->sampler_units_explicit_by_stage[shader_stage]
                                                        [provisional_slot];
            MGLStorageImageBindInput in = {0};
            mglBindingTextureFillStorageImageInput(
                &in, pass,
                0, /* no skip recipe (see the sampled-texture path) */
                resource ? 1 : 0, resource ? resource->binding : 0u, element,
                fallback_metal, (explicit_unit || resource) ? 1 : 0,
                explicit_unit ? 1 : 0,
                explicit_unit ? (uint32_t)program->sampler_units_by_stage
                                                    [shader_stage]
                                                    [provisional_slot]
                              : 0u,
                resource ? resource->sampler_unit : -1,
                resource ? resource->gl_binding : 0u,
                (GLuint)mglRendererGetProgramGLBinding(
                    areas.ctx, shader_stage, _STORAGE_IMAGE_RES, (int)i),
                TEXTURE_UNITS);
            MGLStorageImageBindPlan plan = {0};
            if (mglBindingTexturePlanStorageImage(&in, &plan) != 0 ||
                plan.action == MGL_SI_ACTION_SKIP) {
                continue;
            }
            Texture *ptr = plan.gl_unit < TEXTURE_UNITS
                               ? mglSiState(&areas)->image_units[plan.gl_unit].tex
                               : NULL;
            if (plan.action == MGL_SI_ACTION_ENSURE_TEX) {
                if (ptr) {
                    RETURN_FALSE_ON_FAILURE(mglRendererBindMTLTexture(renderer, ptr));
                }
                continue;
            }
            void *texture = NULL;
            if (ptr) {
                if (mglSiAbortTBindIfEncoderClosed(&areas)) {
                    return false;
                }
                texture = ptr->mtl_data;
                texture = mglSiCreateStorageImageView(
                    texture, &mglSiState(&areas)->image_units[plan.gl_unit]);
            }
            if (!mglSiQueueResourceBinding(
                    use_resource_snapshot, binding_owner,
                    mglSiRenderEncoderOwner(&areas),
                    &resource_snapshot, metal_bind_stage,
                    MGL_RENDER_RESOURCE_BINDING_TEXTURE, texture,
                    plan.metal_slot)) {
                return false;
            }
        }
        if (pass == MGL_SI_PASS_ENSURE &&
            mglRenderEncoderOwnerHasCurrent(mglSiRenderEncoderOwner(&areas)) ==
                0) {
            RETURN_FALSE_ON_FAILURE(
                mglRendererRestoreRenderEncoderAfterTextureUploadPort(renderer,
                                                                       restore_tag));
        }
    }
    if (use_resource_snapshot &&
        !mglSiFlushResourceBindings(binding_owner,
                                    mglSiRenderEncoderOwner(&areas),
                                    &resource_snapshot)) {
        return false;
    }
    return true;
}

bool mglBindingStateBindStorageImagesForVertexProgram(void *renderer,
                                                      Program *vertex_program,
                                                      Program *fragment_program)
{
    if (!renderer) {
        return false;
    }

    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    const int vertex_stage = areas.tessellation->nativeTESActive
                                 ? _TESS_EVALUATION_SHADER
                                 : _VERTEX_SHADER;
    if (!mglBindingStateBindStorageImagesForStage(
            renderer, vertex_stage, vertex_program,
            MGL_RENDER_BINDING_STAGE_VERTEX)) {
        return false;
    }
    if (!mglBindingStateBindStorageImagesForStage(
            renderer, _FRAGMENT_SHADER, fragment_program,
            MGL_RENDER_BINDING_STAGE_FRAGMENT)) {
        return false;
    }
    return true;
}
