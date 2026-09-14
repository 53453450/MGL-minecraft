/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_tess_dispatch.c — the TCS dispatch entry moved out of
 * MGLRenderer+Tessellation.m (P0-1, log 126).
 *
 *   -dispatchTessControlShader:program:contract:  -> mglTessDispatchControlShader
 *   -newTCSStageInBufferForContext:...            -> mglTessDispatchNewTCSStageInBuffer (static)
 *
 * Translation rules are the ones the previous cuts settled: `self` becomes the
 * renderer handle, the pass-manager state and the tessellation record arrive
 * through MGLRendererStateAreas, MGL_STATE() becomes the C twin, "[self ...]"
 * becomes the C entry of mgl_renderer_ports.h and the NSMutableArray of
 * temporaries becomes the mglRendererTemporaries handle.
 *
 * Two ownership details are the whole risk of this cut:
 *   - the compute pipeline handle is a +1 that the ARC local
 *     `(__bridge_transfer id)tcsPipeline` released at every exit; the C twin
 *     releases it at the single `done:` label instead;
 *   - the temporaries set is created +1 and released at the same label, after
 *     the plan has been encoded (mglRenderExecuteComputeExecutionPlan encodes
 *     synchronously, so the command buffer holds the buffers from then on).
 * The copy-back list is cleared at that label too: the method cleared it on
 * every path, and the clear (backend list reset + memset) is idempotent.
 */

#include <stdatomic.h>
#include "mgl_render_pass_manager_ops.h" /* mglRenderPassNewCommandBufferLocked */
#include <stdio.h>
#include <string.h>

#include <CoreFoundation/CoreFoundation.h>

#include "mgl_tess_dispatch.h"
#include "mgl_tess_stage_bind.h"         /* stage-binding + texture plan (log 125) */
#include "mgl_tess_texture.h"            /* mglTessEnsureTextureMetalData */
#include "mgl_tess_compute_ops.h"        /* mglTessBindPointSizeParamsToComputeEncoder */
#include "mgl_stage_copy_back.h"
#include "mgl_renderer_ports.h"          /* state areas, command buffer, processBuffer */
#include "mgl_renderer_backend.h"        /* tcs output / patch-out / capture slots */
#include "mgl_render.h"                  /* buffer creation, copy encodings, execute */
#include "mgl_compute_pipeline_cache.h"  /* mglGetOrCreateProgramComputePipeline */
#include "mgl_draw_tess.h"               /* the tess plan predicates */
#include "mgl_air_tess_abi.h"            /* mglRenderFillDefaultTessFactorBuffer */
#include "mgl_vertex_attrib_query.h"     /* mglRendererGetValidatedVAO */
#include "mgl_vertex_attrib_binding.h"   /* mglRendererResolveVertexAttribBinding */
#include "mgl_index_buffer.h"            /* mglPrimitiveRestartIndexForType */
#include "mgl_buffer_slots.h"            /* kMGLPointSizeBufferIndex */
#include "mgl_draw_support.h"            /* rasterization predicates, polygon offset */
#include "mgl_draw_issue.h"              /* mglDrawHostHandleGeometry */
#include "mgl_buffer_map.h"              /* mglRendererUpdateDirtyBuffer, map entries */
#include "mgl_texture_sampler.h"         /* mglTextureCreateSamplerForTexParam */
#include "mgl_metal_ref.h"               /* mglSafeReleaseMetalObj */
#include "mgl_env_flag.h"                /* MGL_TES_VERTEX_TRACE */
#include "mgl_size_constants.h"          /* kMGLMaxBufferSlots */
#include "mgl_thread_affinity.h"         /* MGL_ASSERT_GL_THREAD */
#include "glm_limits.h"                  /* MAX_ATTRIBS, TEXTURE_UNITS */

/* Declared next to their definitions in Objective-C headers a .c file cannot
 * include; repeated here the way mgl_renderer_ports.c repeats its prototypes.
 * mglRendererBuildCurrentVertexAttribBytes returns NSUInteger there, which is
 * the same 64-bit unsigned type as size_t on every target MGL builds for. */
extern Buffer *getElementBuffer(GLMContext ctx);
extern size_t mglRendererBuildCurrentVertexAttribBytes(GLMContext ctx,
                                                       GLuint attribute,
                                                       const VertexAttrib *attrib,
                                                       uint8_t bytes[16]);
/* Declared extern in MGLRenderer+Tessellation.m before that method moved here. */
extern void mglRecordActivePrimitiveQueryDraw(GLMContext ctx,
                                              GLuint64 generated,
                                              GLuint64 written);

/* MGL_STATE() from MGLRenderer_Private.h, in C (the same twin as
 * mgl_compute_bind.c). */
static GLMState *mglTessDispatchState(const MGLRendererStateAreas *areas)
{
    if (areas->core && areas->core->activeState) {
        return areas->core->activeState;
    }
    return areas->ctx ? areas->ctx->active_state : NULL;
}

static void *mglTessDispatchCommandBufferOwner(
    const MGLRendererStateAreas *areas)
{
    return areas->command ? areas->command->currentCommandBufferOwner : NULL;
}

static void *mglTessDispatchRenderEncoderOwner(
    const MGLRendererStateAreas *areas)
{
    return areas->command ? areas->command->currentRenderEncoderOwner : NULL;
}

/* === File-local twins of the Objective-C statics ==========================
 * MGLRenderer+Tessellation.m still owns the `id`-typed statics for the two AIR
 * TES methods that have not moved yet; these are the C twins over the same
 * mglRender* entries. */

/* The .m's MGL_TESS_RESOURCE_STORAGE_SHARED: MTLResourceStorageModeShared. */
#define MGL_TESS_DISPATCH_STORAGE_SHARED 0u

/* +1 buffer, or NULL. */
static void *mglTessDispatchCreateBuffer(size_t length, uint64_t options)
{
    void *buffer = NULL;
    if (mglRenderCreateBuffer((uint64_t)length, options, NULL, &buffer) == 0 &&
        buffer) {
        return buffer;
    }
    return NULL;
}

/* +1 buffer, or NULL. */
static void *mglTessDispatchCreateBufferWithBytes(const void *bytes,
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

static void *mglTessDispatchBufferContents(void *buffer)
{
    void *contents = NULL;
    uint64_t length = 0u;
    return buffer && mglRenderGetBufferContents(buffer, &contents, &length) == 0
               ? contents
               : NULL;
}

static uint64_t mglTessDispatchBufferLength(void *buffer)
{
    MGLRenderBufferInfo info = {0};
    return buffer && mglRenderGetBufferInfo(buffer, &info) == 0 ? info.length
                                                                : 0u;
}

/* +1 default sampler, or NULL. */
static void *mglTessDispatchCreateSampler(void)
{
    void *sampler = NULL;
    if (mglRenderCreateDefaultSampler(&sampler) == 0 && sampler) {
        return sampler;
    }
    return NULL;
}

static void mglTessDispatchSetRenderVertexBuffer(void *render_encoder_owner,
                                                 void *buffer, size_t offset,
                                                 size_t index)
{
    (void)mglRenderSetRenderBufferForOwner(
        render_encoder_owner, buffer, offset, MGL_RENDER_BINDING_STAGE_VERTEX,
        (uint32_t)index);
}

static void mglTessDispatchSetRenderVertexBytes(void *render_encoder_owner,
                                                const void *bytes,
                                                size_t length, size_t index)
{
    (void)mglRenderSetRenderBytesForOwner(
        render_encoder_owner, bytes, length, MGL_RENDER_BINDING_STAGE_VERTEX,
        (uint32_t)index);
}

static void mglTessDispatchSetRenderVertexTexture(void *render_encoder_owner,
                                                  void *texture, size_t index)
{
    (void)mglRenderSetRenderTextureForOwner(
        render_encoder_owner, texture, MGL_RENDER_BINDING_STAGE_VERTEX,
        (uint32_t)index);
}

static void mglTessDispatchSetRenderVertexSampler(void *render_encoder_owner,
                                                  void *sampler, size_t index)
{
    (void)mglRenderSetRenderSamplerForOwner(
        render_encoder_owner, sampler, MGL_RENDER_BINDING_STAGE_VERTEX,
        (uint32_t)index);
}

/* The .m's mglTessPlanBufferOrBind / mglTessAppendComputeResourceOp pair, with
 * the temporaries set as the C handle. */
static bool mglTessDispatchAppendComputeResourceOp(
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

static bool mglTessDispatchPlanBufferOrBind(
    MGLRenderComputeExecutionPlan *plan, void *temporaries, void *buffer,
    size_t offset, size_t index)
{
    return mglTessDispatchAppendComputeResourceOp(plan, temporaries, 0u, buffer,
                                                  offset, index);
}

/* The .m's mglTESXFBVertexStride. */
static size_t mglTESXFBVertexStride(const Program *program)
{
    return (size_t)mglRenderTESXFBVertexStride((const void *)program);
}

/* The .m's mglTessDrawPrimitives (the `encoder` argument was unused). */
static void mglTessDispatchDrawPrimitives(void *render_encoder_owner,
                                          uint32_t type, size_t vertex_start,
                                          size_t vertex_count,
                                          size_t instance_count,
                                          size_t base_instance)
{
    const MGLRenderDrawPlan plan = {
        .kind = MGL_RENDER_DRAW_ARRAY,
        .primitive_type = (uint32_t)type,
        .vertex_start = vertex_start,
        .vertex_count = vertex_count,
        .instance_count = instance_count,
        .base_instance = base_instance,
    };
    (void)mglRenderEncodeDrawForRenderEncoderOwner(render_encoder_owner, &plan,
                                                   NULL, 0);
}

/* The .m's mglRendererReadableBufferBytes. */
static const uint8_t *mglTessDispatchReadableBufferBytes(Buffer *buffer)
{
    if (!buffer) {
        return NULL;
    }
    if (buffer->data.buffer_data &&
        mglRenderCPUPointerUsable((const void *)buffer->data.buffer_data)) {
        return (const uint8_t *)(uintptr_t)buffer->data.buffer_data;
    }
    if (buffer->data.mtl_data) {
        return (const uint8_t *)mglTessDispatchBufferContents(
            buffer->data.mtl_data);
    }
    return NULL;
}

/* The ARC locals this file replaces were implicit retains: an `id` local kept
 * its object alive until the end of its scope, and a `void *` local does not.
 * Everything the methods held that way is registered in the temporaries set,
 * which is released at the single `done:` label:
 *   - a borrowed handle (backend getter, buffer field) is added as-is;
 *   - a freshly created +1 is added and the creation reference dropped, which
 *     is exactly the ARC `__bridge_transfer` local's end-of-scope release.
 * Without this the XFB copy-back read a Metal buffer whose only keep-alive had
 * been the `id xfbCopyDestination` local (log 128). */
static void mglTessDispatchKeepAlive(void *temporaries, void *object)
{
    if (!temporaries || !object) {
        return;
    }
    mglRendererTemporariesAdd(temporaries, object);
}

static void mglTessDispatchAdopt(void *temporaries, void *object)
{
    if (!temporaries || !object) {
        return;
    }
    mglRendererTemporariesAdd(temporaries, object);
    CFRelease((CFTypeRef)object);
}

/* Was -newTCSStageInBufferForContext:program:first:count:indexType:indices:
 * baseVertex:baseInstance:patchVertices:patchCount:outStride:.  Returns the +1
 * stage_in buffer, or NULL. */
static void *mglTessDispatchNewTCSStageInBuffer(
    void *renderer, GLMContext draw_ctx, Program *tcs_program, GLint first,
    GLsizei count, GLenum index_type, const void *indices, GLint base_vertex,
    GLuint base_instance, GLuint patch_vertices, GLuint patch_count,
    size_t *out_stride)
{
    MGL_ASSERT_GL_THREAD();
    if (out_stride) {
        *out_stride = 0u;
    }
    if (!renderer || !draw_ctx || !tcs_program || count <= 0) {
        return NULL;
    }

    if (!tcs_program->modules[_TESS_CONTROL_SHADER].metallib_bytes) {
        return NULL;
    }

    MGLTessTCSStageInPlan stage_plan = {0};
    if (!mglTessPlanTCSStageIn(patch_vertices, patch_count, count,
                               &stage_plan)) {
        return NULL;
    }
    size_t tcs_in_stride = (size_t)stage_plan.stride;
    size_t member_count = stage_plan.member_count;
    MGLTessStageInMember members[MAX_ATTRIBS];
    memset(members, 0, sizeof(members));
    members[0] = stage_plan.members[0];
    size_t tcs_in_vertices = (size_t)stage_plan.vertices;

    VertexArray *vao = mglRendererGetValidatedVAO(draw_ctx, "tcs.stage_in");
    if (!vao) {
        return NULL;
    }

    const uint8_t *index_bytes = NULL;
    size_t index_offset = (size_t)(uintptr_t)indices;
    uint32_t restart_index = 0u;
    int primitive_restart = 0;
    if (index_type != 0u) {
        Buffer *ebo = getElementBuffer(draw_ctx);
        if (!ebo || !mglRendererProcessBuffer(renderer, ebo)) {
            fprintf(stderr,
                    "MGL TESS WARNING: TCS indexed stage_in has no readable "
                    "element buffer\n");
            return NULL;
        }
        const uint8_t *ebo_bytes = mglTessDispatchReadableBufferBytes(ebo);
        MGLTessIndexedStageInPlan index_plan = {0};
        if (!ebo_bytes ||
            !mglTessPlanIndexedStageIn((uint32_t)index_type,
                                       (uint64_t)index_offset, (int32_t)count,
                                       ebo->size, &index_plan) ||
            index_plan.status != MGL_TESS_INDEXED_STAGE_IN_OK) {
            fprintf(stderr,
                    "MGL TESS WARNING: TCS indexed stage_in element range OOB "
                    "offset=%lu size=%lld",
                    (unsigned long)index_offset, (long long)ebo->size);
            return NULL;
        }
        index_bytes = ebo_bytes + index_offset;
        primitive_restart = mglPrimitiveRestartIndexForType(
            draw_ctx, index_type, &restart_index);
    }

    size_t tcs_in_size = (size_t)stage_plan.bytes;
    void *stage_in_buffer = mglTessDispatchCreateBuffer(
        tcs_in_size, MGL_TESS_DISPATCH_STORAGE_SHARED);
    void *stage_in_contents = mglTessDispatchBufferContents(stage_in_buffer);
    if (!stage_in_contents) {
        return NULL;
    }
    if (!mglTessInitStageInDefaults(stage_in_contents, tcs_in_vertices,
                                    tcs_in_stride)) {
        return NULL;
    }

    if (mglTessTCSStageInEmptyOK((uint32_t)member_count)) {
        if (out_stride) {
            *out_stride = tcs_in_stride;
        }
        return stage_in_buffer;
    }

    MGLTessStageInAttribSrc srcs[MAX_ATTRIBS];
    memset(srcs, 0, sizeof(srcs));
    for (size_t m = 0; m < member_count; m++) {
        const MGLTessStageInMember *member = &members[m];
        if (!mglTessStageInAttribInRange(member->attribute)) {
            continue;
        }
        const VertexAttrib *attrib = &vao->attrib[member->attribute];
        MGLResolvedVertexAttribBinding resolved = {0};
        bool has_binding = mglRendererResolveVertexAttribBinding(
            draw_ctx, vao, member->attribute, "tcs.stage_in", &resolved);
        bool use_current_value = mglTessStageInUseCurrentValue(
                                     vao->enabled_attribs, member->attribute,
                                     has_binding ? 1 : 0) != 0;
        srcs[m].type = attrib->type;
        srcs[m].attrib_size = attrib->size;
        srcs[m].normalized = attrib->normalized;
        if (use_current_value) {
            srcs[m].use_current = 1u;
            if (mglRendererBuildCurrentVertexAttribBytes(
                    draw_ctx, member->attribute, attrib, srcs[m].current) > 0u) {
                srcs[m].current_valid = 1u;
            }
        } else if (has_binding) {
            Buffer *vbo = resolved.buffer;
            if (vbo && mglRendererProcessBuffer(renderer, vbo)) {
                srcs[m].bytes = mglTessDispatchReadableBufferBytes(vbo);
                srcs[m].stride = resolved.stride;
                srcs[m].divisor = resolved.divisor;
                srcs[m].binding_offset = resolved.binding_offset;
                srcs[m].relativeoffset = resolved.relativeoffset;
                srcs[m].buffer_size = mglRenderBufferSizeOrZero(vbo->size);
            }
        }
    }
    if (!mglTessPackStageInRecords(
            stage_in_contents, tcs_in_vertices, tcs_in_stride, first, count,
            index_bytes, index_type, primitive_restart != 0, restart_index,
            base_vertex, base_instance, members, (uint32_t)member_count, srcs)) {
        return NULL;
    }

    if (out_stride) {
        *out_stride = tcs_in_stride;
    }
    return stage_in_buffer;
}

/* Was -dispatchTessControlShader:program:contract:. */
bool mglTessDispatchControlShader(void *renderer, GLMContext glm_ctx,
                                  Program *tcs_program,
                                  const MGLAIRTessDrawContract *contract)
{
    MGLStageBindingCopyBackList stage_copy_backs = {0};
    void *tcs_pipeline = NULL;
    void *temporaries = NULL;
    bool ok = false;

    if (!renderer || !tcs_program || !glm_ctx || !contract) {
        return false;
    }

    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    Shader *tcs_shader = tcs_program->shader_slots[_TESS_CONTROL_SHADER];
    if (!mglTessStageHasCompiledFunction(
            tcs_shader ? 1 : 0,
            tcs_program->modules[_TESS_CONTROL_SHADER].mtl_function ? 1 : 0)) {
        fprintf(stderr,
                "MGL TESS WARNING: TCS program %u has no compiled function",
                tcs_program->name);
        return false;
    }

    /* Create compute pipeline state for TCS kernel.  The ready handle is a +1
     * the ARC local owned; here the `done:` label releases it. */
    void *tcs_pipeline_handle = NULL;
    char tcs_pipeline_error[512] = {0};
    int tcs_pipeline_result = mglGetOrCreateProgramComputePipeline(
        tcs_program, _TESS_CONTROL_SHADER, &tcs_pipeline_handle,
        tcs_pipeline_error, sizeof(tcs_pipeline_error));
    if (mglTessComputePipelineReady(tcs_pipeline_result,
                                    tcs_pipeline_handle ? 1 : 0)) {
        tcs_pipeline = tcs_pipeline_handle;
    }
    if (!tcs_pipeline) {
        fprintf(stderr,
                "MGL TESS ERROR: failed to create TCS compute pipeline for "
                "program %u: %s",
                tcs_program->name,
                tcs_pipeline_error[0] ? tcs_pipeline_error : "unknown error");
        return false;
    }

    temporaries = mglRendererTemporariesCreate();

    /* PASS 1: Pre-resolve all Metal textures that the TCS kernel needs.
     * This must happen BEFORE we open a compute encoder, because lazy
     * Metal texture creation (bindMTLTexture:) may open its own blit
     * encoder on the command buffer, and Metal forbids two encoders
     * on the same command buffer simultaneously.  End any active render
     * encoder first for the same reason. */
    if (mglTessMustEndRenderBeforeCompute(mglRenderEncoderOwnerHasCurrent(
            mglTessDispatchRenderEncoderOwner(&areas)))) {
        mglRendererEndRenderEncodingPort(renderer);
    }

    /* Ensure a writable command buffer exists.  The GL_PATCHES path returns
     * before processGLState() (which normally creates the command buffer),
     * and prior operations (glBufferData, glEndQuery, etc.) may have
     * committed the previous command buffer. */
    MGLRenderCommandBufferState command_state = {0};
    const int has_command_state = mglRenderCommandBufferOwnerHasState(
        mglTessDispatchCommandBufferOwner(&areas), &command_state);
    if (mglTessCommandBufferNeedsNew(has_command_state, command_state.status)) {
        /* -newCommandBuffer was METAL_LOCK + -newCommandBufferLocked +
         * METAL_UNLOCK; the lock is the GL-thread assertion. */
        MGL_ASSERT_GL_THREAD();
        if (!mglRenderPassNewCommandBufferLocked(renderer)) {
            fprintf(stderr,
                    "MGL TESS ERROR: failed to create command buffer for TCS "
                    "dispatch\n");
            goto done;
        }
    }

    MGLTessTextureBind tcs_texture_binds[TEXTURE_UNITS * 2u];
    const uint32_t tcs_texture_bind_count = mglTessCollectTextureBinds(
        glm_ctx, tcs_program, _TESS_CONTROL_SHADER, tcs_texture_binds,
        (uint32_t)(sizeof(tcs_texture_binds) / sizeof(tcs_texture_binds[0])));
    if (!mglTessEnsureTextureMetalData(renderer, tcs_texture_binds,
                                       tcs_texture_bind_count, glm_ctx)) {
        goto done;
    }

    MGLTessStageBufferBindingList stage_buffer_bindings = {0};
    if (!mglTessPrepareStageBufferBindings(renderer, &stage_buffer_bindings,
                                           _TESS_CONTROL_SHADER,
                                           &stage_copy_backs)) {
        goto done;
    }

    MGLRenderComputeExecutionPlan execution_plan = {0};
    execution_plan.pipeline = tcs_pipeline;

    if (!mglTessPlanTextureBinds(renderer, tcs_texture_binds,
                                 tcs_texture_bind_count, glm_ctx,
                                 &execution_plan, temporaries)) {
        goto done;
    }

    /* Bind stage buffers (UBO, SSBO, atomic counters) for TCS. */
    if (!mglTessBindPreparedStageBufferBindings(&stage_buffer_bindings, NULL,
                                                &execution_plan, temporaries)) {
        goto done;
    }
    mglTessBindPointSizeParamsToComputeEncoder(renderer, tcs_program,
                                               _TESS_CONTROL_SHADER,
                                               &execution_plan, temporaries);

    MGLTessTCSCoreLayout tcs_layout;
    if (!mglTessComputeTCSCoreLayout(tcs_program, contract, &tcs_layout)) {
        goto done;
    }
    areas.tessellation->tcsOutputStride = tcs_layout.output_stride;
    areas.tessellation->tcsOutVertices = tcs_layout.tcs_out_vertices;
    const GLuint patch_vertices = tcs_layout.patch_vertices;
    const GLuint instance_count = tcs_layout.instance_count;
    const GLuint patch_count = tcs_layout.patch_count;

    void *tcs_output_buffer = mglTessDispatchCreateBuffer(
        (size_t)tcs_layout.output_bytes, MGL_TESS_DISPATCH_STORAGE_SHARED);
    (void)mglRendererBackendSetTcsOutputBuffer(areas.backend, tcs_output_buffer);
    void *tcs_output_contents =
        mglTessDispatchBufferContents(tcs_output_buffer);
    if (!tcs_output_contents) {
        goto done;
    }
    memset(tcs_output_contents, 0, (size_t)tcs_layout.output_bytes);
    areas.tessellation->tcsOutputOffset = 0u;
    mglTessDispatchAdopt(temporaries, tcs_output_buffer);

    void *tcs_patch_out_buffer = mglTessDispatchCreateBuffer(
        (size_t)tcs_layout.patch_out_bytes, MGL_TESS_DISPATCH_STORAGE_SHARED);
    (void)mglRendererBackendSetTcsPatchOutBuffer(areas.backend,
                                                 tcs_patch_out_buffer);
    void *tcs_patch_out_contents =
        mglTessDispatchBufferContents(tcs_patch_out_buffer);
    if (!tcs_patch_out_contents) {
        goto done;
    }
    memset(tcs_patch_out_contents, 0, (size_t)tcs_layout.patch_out_bytes);
    mglTessDispatchAdopt(temporaries, tcs_patch_out_buffer);

    GLuint indirect_params[2] = {0u, 0u};
    mglTessFillTCSIndirectParams(patch_vertices, instance_count,
                                 indirect_params);
    void *indirect_buffer = mglTessDispatchCreateBufferWithBytes(
        indirect_params, sizeof(indirect_params),
        MGL_TESS_DISPATCH_STORAGE_SHARED);
    if (!indirect_buffer) {
        goto done;
    }
    mglTessDispatchAdopt(temporaries, indirect_buffer);

    void *tess_factor_buffer = mglTessDispatchCreateBuffer(
        (size_t)tcs_layout.factor_bytes, MGL_TESS_DISPATCH_STORAGE_SHARED);
    void *tess_factor_contents =
        mglTessDispatchBufferContents(tess_factor_buffer);
    if (!tess_factor_contents) {
        goto done;
    }
    /* GL 4.6 §11.2.2: TCS-unwritten tess levels take PATCH_DEFAULT_*. */
    if (mglRenderFillDefaultTessFactorBuffer(
            tess_factor_contents, tcs_layout.factor_bytes,
            mglTessDispatchState(&areas)->var.patch_default_outer_level,
            mglTessDispatchState(&areas)->var.patch_default_inner_level,
            tcs_layout.patch_count) != 0) {
        goto done;
    }
    mglTessDispatchAdopt(temporaries, tess_factor_buffer);

    size_t tcs_in_stride = 0u;
    void *tcs_stage_in_buffer =
        mglRendererBackendGetTessVertexCaptureBuffer(areas.backend);
    size_t tcs_stage_in_offset =
        areas.tessellation->tessVertexCaptureOffset;
    MGLTessTCSStageInSourcePlan stage_in_source = {0};
    mglTessPlanTCSStageInSource(tcs_stage_in_buffer ? 1 : 0, tcs_program,
                                &stage_in_source);
    if (stage_in_source.kind == MGL_TESS_TCS_STAGE_IN_CAPTURE) {
        tcs_in_stride = (size_t)stage_in_source.stride;
        mglRendererTemporariesAdd(temporaries, tcs_stage_in_buffer);
    } else {
        tcs_stage_in_buffer = mglTessDispatchNewTCSStageInBuffer(
            renderer, glm_ctx, tcs_program, contract->first,
            (GLsizei)contract->vertex_count, contract->index_type,
            (const void *)(uintptr_t)contract->index_source,
            contract->base_vertex, contract->base_instance, patch_vertices,
            patch_count, &tcs_in_stride);
        tcs_stage_in_offset = 0u;
        if (tcs_stage_in_buffer) {
            mglRendererTemporariesAdd(temporaries, tcs_stage_in_buffer);
        }
    }
    if (!tcs_stage_in_buffer) {
        fprintf(stderr,
                "MGL TESS WARNING: failed to pack TCS stage_in buffer for "
                "program %u",
                tcs_program ? (unsigned)tcs_program->name : 0u);
        goto done;
    }
    if (!mglTessAppendTCSCoreBindings(
            &execution_plan, tcs_output_buffer, tcs_patch_out_buffer,
            indirect_buffer, tess_factor_buffer, tcs_stage_in_buffer,
            (uint64_t)tcs_stage_in_offset, &tcs_layout)) {
        goto done;
    }

    {
        MGLRenderCopyBackEntry copy_back_entries[kMGLMaxBufferSlots] = {0};
        uint32_t copy_back_entry_count = mglRenderCollectCopyBackEntries(
            (const MGLRenderCopyBackEntry *)stage_copy_backs.slots,
            kMGLMaxBufferSlots, copy_back_entries, kMGLMaxBufferSlots);
        execution_plan.barrier_scope = MGL_RENDER_COMPUTE_BARRIER_BUFFERS;
        MGLRenderComputeExecutionResult execution_result = {0};
        char execution_error[256] = {0};
        if (mglRenderExecuteComputeExecutionPlan(
                mglTessDispatchCommandBufferOwner(&areas),
                areas.gpu_recovery_command_owner
                    ? *areas.gpu_recovery_command_owner
                    : NULL,
                &execution_plan, copy_back_entries, copy_back_entry_count, 1u,
                &execution_result, execution_error,
                sizeof(execution_error)) != 0) {
            if (execution_result.transaction.device_reset_requested) {
                atomic_store_explicit(&areas.core->deviceResetRequested, true,
                                      memory_order_release);
            }
            fprintf(stderr, "MGL TESS ERROR: C++ TCS execution failed: %s",
                    execution_error[0] ? execution_error : "unknown error");
            goto done;
        }
    }

    /* Save tess factor buffer for TES patch-draw path. */
    (void)mglRendererBackendSetCurrentTessFactorBuffer(areas.backend,
                                                       tess_factor_buffer);
    ok = true;

done:
    mglClearStageBindingCopyBacks(renderer, &stage_copy_backs);
    if (temporaries) {
        mglRendererTemporariesRelease(temporaries);
    }
    CFRelease((CFTypeRef)tcs_pipeline);
    return ok;
}

/* === AIR TES as a render vertex function (log 127) ========================
 * Was -dispatchAIRTessEvalVertexRender:program:contract:patchCount:
 * instanceCount:baseInstance:.  The CPU domain expansion seeds TessCoord
 * records once, then the render encoder replays a per-patch drawPrimitives
 * with the TES compiled as the vertex stage, which removes the per-patch TES
 * compute dispatch and the compute→render encoder switch of the compute
 * expansion path. */

/* The method's function-static "logged once" flag. */
static int s_tes_vertex_multi_instance_logged = 0;

bool mglTessDispatchAIRTessEvalVertexRender(
    void *renderer, GLMContext glm_ctx, Program *tes_program,
    const MGLAIRTessDrawContract *contract, GLuint patch_count,
    GLsizei instance_count, GLuint base_instance)
{
    void *temporaries = NULL;
    bool ok = false;
    MGL_ASSERT_GL_THREAD();
    if (!renderer || !tes_program || !glm_ctx || !contract ||
        patch_count == 0u || instance_count <= 0) {
        return false;
    }

    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    /* ARC kept every `id` local of this method alive to the end of the scope;
     * the temporaries set does that here (see mglTessDispatchKeepAlive). */
    temporaries = mglRendererTemporariesCreate();

    /* This draw takes the render-vertex path: the TES stage binds its
     * resources read-only into the render encoder (no isolated copies). */
    areas.tessellation->tessVertexRenderActive = 1;
    Shader *tes_shader = tes_program->shader_slots[_TESS_EVALUATION_SHADER];
    if (!mglTessStageHasCompiledFunction(
            tes_shader ? 1 : 0,
            tes_program->modules[_TESS_EVALUATION_SHADER].mtl_function ? 1 : 0)) {
        fprintf(stderr,
                "MGL TESS ERROR: TES-vertex program %u has no compiled function",
                (unsigned)tes_program->name);
        ok = false;
        goto done;
    }

    void *tcs_output_buffer =
        mglRendererBackendGetTcsOutputBuffer(areas.backend);
    void *tess_factor_buffer =
        mglRendererBackendGetCurrentTessFactorBuffer(areas.backend);
    void *capture_buffer =
        mglRendererBackendGetTessVertexCaptureBuffer(areas.backend);
    mglTessDispatchKeepAlive(temporaries, tcs_output_buffer);
    mglTessDispatchKeepAlive(temporaries, tess_factor_buffer);
    mglTessDispatchKeepAlive(temporaries, capture_buffer);
    MGLTessEvalGlInPlan gl_in_plan = {0};
    if (!mglTessResolveEvalGlIn(
            contract, tcs_output_buffer ? 1 : 0,
            (uint64_t)areas.tessellation->tcsOutputOffset,
            (uint64_t)areas.tessellation->tcsOutputStride,
            areas.tessellation->tcsOutVertices, capture_buffer ? 1 : 0,
            (uint64_t)areas.tessellation->tessVertexCaptureOffset,
            areas.tessellation->tessIndexedDraw ? 1 : 0,
            (uint32_t)areas.tessellation->tessInstanceRecords,
            (uint32_t)instance_count, &gl_in_plan)) {
        fprintf(stderr, "MGL TESS ERROR: missing TES-vertex inputs program=%u",
                (unsigned)tes_program->name);
        ok = false;
        goto done;
    }
    void *gl_in_buffer = gl_in_plan.from_tcs ? tcs_output_buffer : capture_buffer;
    const size_t gl_in_offset = (size_t)gl_in_plan.gl_in_offset;
    const size_t gl_in_instance_stride =
        (size_t)gl_in_plan.gl_in_instance_stride;
    const GLuint gl_in_vertices = gl_in_plan.gl_in_vertices;
    if (!mglTessEvalInputsReady(gl_in_buffer ? 1 : 0,
                                tess_factor_buffer ? 1 : 0)) {
        fprintf(stderr, "MGL TESS ERROR: missing TES-vertex inputs program=%u",
                (unsigned)tes_program->name);
        ok = false;
        goto done;
    }
    if (mglTessMultiInstanceTCSReuseWarn(gl_in_plan.from_tcs ? 1 : 0,
                                         (int32_t)instance_count)) {
        if (!s_tes_vertex_multi_instance_logged) {
            fprintf(stderr,
                    "MGL TESS ERROR: multi-instance TES-vertex with TCS reuses "
                    "instance-0 control points (program=%u instances=%d)",
                    (unsigned)tes_program->name, (int)instance_count);
            s_tes_vertex_multi_instance_logged = 1;
        }
        if (mglTessMultiInstanceTCSReuseIsError(gl_in_plan.from_tcs ? 1 : 0,
                                                (int32_t)instance_count)) {
            ok = false;
            goto done;
        }
    }

    const uint16_t *factor_bytes =
        (const uint16_t *)mglTessDispatchBufferContents(tess_factor_buffer);
    MGLTessEvalComputePlan eval_plan = {0};
    if (!mglTessPlanEvalCompute(tes_program, factor_bytes,
                                mglTessDispatchBufferLength(tess_factor_buffer),
                                patch_count, (uint32_t)instance_count,
                                &eval_plan)) {
        fprintf(stderr, "MGL TESS ERROR: TES-vertex plan failed program=%u",
                (unsigned)tes_program->name);
        ok = false;
        goto done;
    }
    if (eval_plan.empty) {
        ok = true;
        goto done;
    }
    const GLuint eval_instance_count = eval_plan.instance_count;
    const GLuint items_per_instance = eval_plan.items_per_instance;
    const size_t out_stride = eval_plan.out_stride;
    const size_t out_size = (size_t)eval_plan.out_size;

    void *out_buffer = mglTessDispatchCreateBuffer(
        out_size, MGL_TESS_DISPATCH_STORAGE_SHARED);
    mglTessDispatchAdopt(temporaries, out_buffer);
    void *out_contents = mglTessDispatchBufferContents(out_buffer);
    if (!out_contents) {
        fprintf(stderr,
                "MGL TESS ERROR: failed to allocate TES-vertex domain stream "
                "(%lu bytes) program=%u",
                (unsigned long)out_size, (unsigned)tes_program->name);
        ok = false;
        goto done;
    }
    if (mglTessSeedEvalOutputRecords(tes_program, factor_bytes, patch_count,
                                     eval_instance_count, out_contents, out_size,
                                     (uint32_t)out_stride) !=
        items_per_instance) {
        fprintf(stderr,
                "MGL TESS ERROR: TES-vertex domain seed failed program=%u",
                (unsigned)tes_program->name);
        ok = false;
        goto done;
    }

    /* The render-vertex path never writes its resources; only the isolated
     * binding *initialization* copy is relevant, and that lands on the command
     * buffer before the render pass.  Since the copy is flushed eagerly, the
     * copy-back list is discarded. */
    MGLTessStageBufferBindingList stage_buffer_bindings = {0};
    MGLStageBindingCopyBackList stage_copy_backs = {0};
    if (!mglTessPrepareStageBufferBindings(renderer, &stage_buffer_bindings,
                                           _TESS_EVALUATION_SHADER,
                                           &stage_copy_backs)) {
        mglClearStageBindingCopyBacks(renderer, &stage_copy_backs);
        ok = false;
        goto done;
    }
    mglClearStageBindingCopyBacks(renderer, &stage_copy_backs);

    MGLTessTextureBind tes_texture_binds[TEXTURE_UNITS * 2u];
    const uint32_t tes_texture_bind_count = mglTessCollectTextureBinds(
        glm_ctx, tes_program, _TESS_EVALUATION_SHADER, tes_texture_binds,
        (uint32_t)(sizeof(tes_texture_binds) / sizeof(tes_texture_binds[0])));
    if (!mglTessEnsureTextureMetalData(renderer, tes_texture_binds,
                                       tes_texture_bind_count, glm_ctx)) {
        ok = false;
        goto done;
    }

    const GLenum tess_raster_mode = mglTessRasterGLMode(tes_program);
    MGLTessRasterQueryPlan query = {0};
    mglTessPlanRasterQuery(tes_program, (uint64_t)instance_count,
                           (uint64_t)items_per_instance, 0, 0u, 0u, &query);
    if (mglTessDispatchState(&areas)->caps.rasterizer_discard) {
        areas.batching->currentCommandBufferHasWork = 1;
        mglRecordActivePrimitiveQueryDraw(glm_ctx, query.prims, query.written);
        ok = true;
        goto done;
    }

    areas.tessellation->tessComputeActive = 1;
    areas.tessellation->tessComputeProgram = tes_program;
    const int state_ready = mglRenderPassProcessGLStateLocked(renderer, 1);
    if (!mglTessPassthroughRasterReady(
            state_ready ? 1 : 0,
            mglRenderEncoderOwnerHasCurrent(
                mglTessDispatchRenderEncoderOwner(&areas)),
            mglDrawRasterizationIsEmpty(renderer) ? 1 : 0)) {
        fprintf(stderr, "MGL TESS ERROR: TES-vertex raster skip program=%u",
                (unsigned)tes_program->name);
        areas.tessellation->tessComputeActive = 0;
        areas.tessellation->tessComputeProgram = NULL;
        ok = false;
        goto done;
    }
    MGLTessEvalVertexPatch *patches = (MGLTessEvalVertexPatch *)calloc(
        patch_count, sizeof(MGLTessEvalVertexPatch));
    uint32_t *contracts =
        (uint32_t *)calloc(patch_count, 4u * sizeof(uint32_t));
    if (!patches || !contracts) {
        free(patches);
        free(contracts);
        areas.tessellation->tessComputeActive = 0;
        areas.tessellation->tessComputeProgram = NULL;
        ok = false;
        goto done;
    }
    const uint32_t live_patches = mglTessBuildEvalVertexPatches(
        tes_program, factor_bytes, patch_count, patches, contracts);
    for (uint32_t p = 0; p < live_patches; p++) {
        contracts[p * 4u + 1u] = gl_in_vertices;
    }
    if (mgl_env_flag_enabled("MGL_TES_VERTEX_TRACE")) {
        fprintf(stderr,
                "MGL TESS-vertex draw program=%u patches=%u live=%u "
                "itemsPerInstance=%u instances=%d point=%d",
                (unsigned)tes_program->name, (unsigned)patch_count,
                (unsigned)live_patches, (unsigned)items_per_instance,
                (int)instance_count,
                (int)(tes_program->tess_gen_point_mode != 0));
    }
    mglDrawApplyPolygonOffset(renderer, tess_raster_mode);
    void *owner = mglTessDispatchRenderEncoderOwner(&areas);

    mglTessBindStageBufferBindingsToRenderEncoderOwner(owner,
                                                       &stage_buffer_bindings);
    if (tes_program->uses_point_size_params) {
        float point_size_params[2] = {0.f, 0.f};
        mglTessFillPointSizeParams(
            mglTessDispatchState(&areas)->var.point_size > 0.0f
                ? mglTessDispatchState(&areas)->var.point_size
                : 0.0f,
            mglTessDispatchState(&areas)->caps.program_point_size ? 1 : 0,
            point_size_params);
        mglTessDispatchSetRenderVertexBytes(owner, point_size_params,
                                            sizeof(point_size_params),
                                            kMGLPointSizeBufferIndex);
    }
    for (uint32_t i = 0; i < tes_texture_bind_count; i++) {
        const MGLTessTextureBind *bind = &tes_texture_binds[i];
        void *texture = NULL;
        Texture *ptr = NULL;
        if (mglTessTextureBindIsStorage(bind->kind)) {
            ptr = mglTessDispatchState(&areas)->image_units[bind->gl_unit].tex;
            if (ptr) {
                texture = ptr->mtl_data;
                texture = mglRendererStorageImageTexture(
                    texture,
                    &mglTessDispatchState(&areas)->image_units[bind->gl_unit]);
            }
        } else {
            ptr = mglTessDispatchState(&areas)->active_textures[bind->gl_unit];
            texture = ptr ? ptr->mtl_data : NULL;
        }
        mglTessDispatchSetRenderVertexTexture(owner, texture,
                                              bind->metal_slot);
        if (!mglTessTextureBindNeedsSampler(bind->kind,
                                            bind->combined_sampler_slot)) {
            continue;
        }
        void *sampler = NULL;
        int created_sampler = 0;
        if (mglTessDispatchState(&areas)->texture_samplers[bind->gl_unit]) {
            Sampler *gl_sampler =
                mglTessDispatchState(&areas)->texture_samplers[bind->gl_unit];
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
            sampler = mglTessDispatchCreateSampler();
            created_sampler = 1;
        }
        if (sampler) {
            mglTessDispatchSetRenderVertexSampler(
                owner, sampler, bind->combined_sampler_slot);
        }
        if (created_sampler) {
            /* The encoder has been told about it; the ARC local released its
             * +1 at the end of the same iteration. */
            CFRelease((CFTypeRef)sampler);
        }
    }

    void *patch_inputs = mglRendererBackendGetTcsPatchOutBuffer(areas.backend);
    mglTessDispatchKeepAlive(temporaries, patch_inputs);
    const uint32_t prim_type = mglTessRasterPrimitiveType(tes_program);

    for (GLsizei i = 0; i < instance_count; i++) {
        const size_t seed_offset = (size_t)mglTessPassthroughInstanceOffset(
            (uint32_t)i, items_per_instance, (uint32_t)out_stride);
        mglTessDispatchSetRenderVertexBuffer(owner, out_buffer, seed_offset,
                                             MGL_AIR_TESS_SLOT_TCS_OUTPUT);
        if (tes_program->tess_cull_distance_count > 0u) {
            /* Cull partner read reuses the seed record stream at slot 28. */
            mglTessDispatchSetRenderVertexBuffer(owner, out_buffer, seed_offset,
                                                 28u);
        }
        mglTessDispatchSetRenderVertexBuffer(
            owner, gl_in_buffer,
            gl_in_offset + (size_t)i * gl_in_instance_stride,
            MGL_AIR_TESS_SLOT_GL_IN);
        for (uint32_t p = 0; p < live_patches; p++) {
            mglTessDispatchSetRenderVertexBuffer(
                owner, tess_factor_buffer,
                (size_t)contracts[p * 4u + 0u] *
                    MGL_AIR_TESS_FACTOR_RECORD_BYTES,
                MGL_AIR_TESS_SLOT_TESS_FACTOR);
            if (patch_inputs) {
                mglTessDispatchSetRenderVertexBuffer(
                    owner, patch_inputs,
                    (size_t)(contracts[p * 4u + 0u]) *
                        contract->patch_out_stride,
                    MGL_AIR_TESS_SLOT_PATCH_OUT);
            }
            mglTessDispatchSetRenderVertexBytes(owner, &contracts[p * 4u],
                                                4u * sizeof(uint32_t),
                                                MGL_AIR_TESS_SLOT_INDIRECT);
            mglTessDispatchDrawPrimitives(owner, prim_type,
                                          (size_t)patches[p].base,
                                          (size_t)patches[p].items, 1u,
                                          (size_t)base_instance + (size_t)i);
        }
    }
    free(patches);
    free(contracts);
    areas.batching->currentCommandBufferHasWork = 1;
    mglRecordActivePrimitiveQueryDraw(glm_ctx, query.prims, query.written);
    areas.tessellation->tessComputeActive = 0;
    areas.tessellation->tessComputeProgram = NULL;
    ok = true;
    goto done;
done:
    /* The set releases every handle this method kept alive, plus its own
     * reference (the ARC scope-exit releases). */
    if (temporaries) {
        mglRendererTemporariesRelease(temporaries);
    }
    return ok;
}

/* === AIR TES as a compute expansion (log 128) =============================
 * Was -dispatchAIRTessEvalCompute:program:contract:patchCount:instanceCount:
 * baseInstance:.  Isolines / point-mode TES expands one vertex record per work
 * item with the AIR TES compute kernel (backend ABI: stage_in(24) factors(26)
 * patchInputs(27) stageOut(28) indirect(29)), then rasterizes through the
 * passthrough vertex stage as lines / points.  Each patch owns a contiguous
 * item span; per-patch item counts differ, so the runtime dispatches per patch
 * with the patch id and output base in the contract buffer (slot 29). */

/* The method's function-static "logged once" flag. */
static int s_multi_instance_tcs_logged = 0;

bool mglTessDispatchAIRTessEvalCompute(
    void *renderer, GLMContext glm_ctx, Program *tes_program,
    const MGLAIRTessDrawContract *contract, GLuint patch_count,
    GLsizei instance_count, GLuint base_instance)
{
    /* The three objects the method's ARC locals owned; the single `done:`
     * label below releases them on every exit path. */
    void *tes_pipeline = NULL;
    void *temporaries = NULL;
    MGLStageBindingCopyBackList stage_copy_backs = {0};
    bool ok = false;

    if (!renderer || !tes_program || !glm_ctx || !contract ||
        patch_count == 0u || instance_count <= 0) {
        return false;
    }

    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    /* This draw takes the compute expansion path: the TES stage needs
     * isolated bindings and copy-backs (the kernel writes its outputs), even
     * when the program also carries the render-vertex function. */
    areas.tessellation->tessVertexRenderActive = 0;

    Shader *tes_shader = tes_program->shader_slots[_TESS_EVALUATION_SHADER];
    if (!mglTessStageHasCompiledFunction(
            tes_shader ? 1 : 0,
            tes_program->modules[_TESS_EVALUATION_SHADER].mtl_function ? 1 : 0)) {
        fprintf(stderr,
                "MGL TESS WARNING: TES program %u has no compiled function",
                tes_program->name);
        return false;
    }

    void *tes_pipeline_handle = NULL;
    char tes_pipeline_error[512] = {0};
    int tes_pipeline_result = mglGetOrCreateProgramComputePipeline(
        tes_program, _TESS_EVALUATION_SHADER, &tes_pipeline_handle,
        tes_pipeline_error, sizeof(tes_pipeline_error));
    if (mglTessComputePipelineReady(tes_pipeline_result,
                                    tes_pipeline_handle ? 1 : 0)) {
        tes_pipeline = tes_pipeline_handle;
    }
    if (!tes_pipeline) {
        fprintf(stderr,
                "MGL TESS ERROR: failed to create TES compute pipeline for "
                "program %u: %s",
                tes_program->name,
                tes_pipeline_error[0] ? tes_pipeline_error : "unknown error");
        return false;
    }

    /* The ARC locals of this method kept their objects alive until the end of
     * the scope; the set does that here (see mglTessDispatchKeepAlive). */
    temporaries = mglRendererTemporariesCreate();

    /* Inputs: gl_in is the post-TCS control point stream (or the VS capture
     * when there is no TCS, which the draw path already aliased into
     * tcsOutputBuffer).  Factors and per-patch inputs come from the TCS
     * dispatch (or defaults). */
    void *tcs_output_buffer =
        mglRendererBackendGetTcsOutputBuffer(areas.backend);
    void *tess_factor_buffer =
        mglRendererBackendGetCurrentTessFactorBuffer(areas.backend);
    void *capture_buffer =
        mglRendererBackendGetTessVertexCaptureBuffer(areas.backend);
    mglTessDispatchKeepAlive(temporaries, tcs_output_buffer);
    mglTessDispatchKeepAlive(temporaries, tess_factor_buffer);
    mglTessDispatchKeepAlive(temporaries, capture_buffer);
    MGLTessEvalGlInPlan gl_in_plan = {0};
    if (!mglTessResolveEvalGlIn(
            contract, tcs_output_buffer ? 1 : 0,
            (uint64_t)areas.tessellation->tcsOutputOffset,
            (uint64_t)areas.tessellation->tcsOutputStride,
            areas.tessellation->tcsOutVertices, capture_buffer ? 1 : 0,
            (uint64_t)areas.tessellation->tessVertexCaptureOffset,
            areas.tessellation->tessIndexedDraw ? 1 : 0,
            (uint32_t)areas.tessellation->tessInstanceRecords,
            (uint32_t)instance_count, &gl_in_plan)) {
        fprintf(stderr, "MGL TESS ERROR: missing TES compute inputs program=%u",
                (unsigned)tes_program->name);
        goto done;
    }
    void *gl_in_buffer = gl_in_plan.from_tcs ? tcs_output_buffer : capture_buffer;
    size_t gl_in_offset = (size_t)gl_in_plan.gl_in_offset;
    size_t gl_in_stride = (size_t)gl_in_plan.gl_in_stride;
    GLuint gl_in_vertices = gl_in_plan.gl_in_vertices;
    if (!mglTessEvalInputsReady(gl_in_buffer ? 1 : 0,
                                tess_factor_buffer ? 1 : 0)) {
        fprintf(stderr, "MGL TESS ERROR: missing TES compute inputs program=%u",
                (unsigned)tes_program->name);
        goto done;
    }
    void *control_point_index_buffer =
        mglRendererBackendGetTessControlPointIndexBuffer(areas.backend);
    mglTessDispatchKeepAlive(temporaries, control_point_index_buffer);
    if (!mglTessEvalIndexedGatherReady(
            areas.tessellation->tessIndexedDraw ? 1 : 0,
            control_point_index_buffer ? 1 : 0,
            (uint32_t)areas.tessellation->tessInstanceRecords)) {
        fprintf(stderr,
                "MGL TESS ERROR: indexed TES compute missing gather "
                "program=%u",
                (unsigned)tes_program->name);
        goto done;
    }
    const int gl_in_from_tcs = gl_in_plan.from_tcs != 0u;
    /* TCS currently expands one instance of control points / factors.
     * TES still loops instances for XFB/output bases.  Reusing instance-0
     * TCS outs is wrong when VS outputs vary by gl_InstanceID.  Until
     * per-instance TCS re-dispatch exists: one-shot log, and hard-fail when
     * MGL_TESS_MULTI_INSTANCE_ERROR is set. */
    if (mglTessMultiInstanceTCSReuseWarn(gl_in_from_tcs, (int32_t)instance_count)) {
        if (!s_multi_instance_tcs_logged) {
            fprintf(stderr,
                    "MGL TESS ERROR: multi-instance TES with TCS reuses "
                    "instance-0 control points (program=%u instances=%d); "
                    "set MGL_TESS_MULTI_INSTANCE_ERROR=1 to fail the draw",
                    (unsigned)tes_program->name, (int)instance_count);
            s_multi_instance_tcs_logged = 1;
        }
        if (mglTessMultiInstanceTCSReuseIsError(gl_in_from_tcs,
                                                (int32_t)instance_count)) {
            goto done;
        }
    }
    const size_t gl_in_instance_stride =
        (size_t)gl_in_plan.gl_in_instance_stride;

    /* Compute per-patch item counts and the per-instance total. */
    const uint16_t *factor_bytes =
        (const uint16_t *)mglTessDispatchBufferContents(tess_factor_buffer);
    MGLTessEvalComputePlan eval_plan = {0};
    if (!mglTessPlanEvalCompute(tes_program, factor_bytes,
                                mglTessDispatchBufferLength(tess_factor_buffer),
                                patch_count, (uint32_t)instance_count,
                                &eval_plan)) {
        fprintf(stderr, "MGL TESS ERROR: TES compute plan failed program=%u",
                (unsigned)tes_program->name);
        goto done;
    }
    if (eval_plan.empty) {
        /* Every patch discarded (outer ≤ 0, e.g. CTS isolines with
         * outer=-1).  Empty expansion is success — do not raise
         * GL_INVALID_OPERATION. */
        ok = true;
        goto done;
    }
    const GLuint eval_instance_count = eval_plan.instance_count;
    const GLuint items_per_instance = eval_plan.items_per_instance;
    size_t out_stride = eval_plan.out_stride;
    const size_t out_size = (size_t)eval_plan.out_size;
    void *out_buffer = mglTessDispatchCreateBuffer(
        out_size, MGL_TESS_DISPATCH_STORAGE_SHARED);
    mglTessDispatchAdopt(temporaries, out_buffer);
    void *out_contents = mglTessDispatchBufferContents(out_buffer);
    if (!out_contents) {
        fprintf(stderr,
                "MGL TESS ERROR: failed to allocate TES compute output "
                "(%lu bytes) program=%u",
                (unsigned long)out_size, (unsigned)tes_program->name);
        goto done;
    }
    if (mglTessSeedEvalOutputRecords(tes_program, factor_bytes, patch_count,
                                     eval_instance_count, out_contents, out_size,
                                     (uint32_t)out_stride) !=
        items_per_instance) {
        fprintf(stderr, "MGL TESS ERROR: TES domain seed failed program=%u",
                (unsigned)tes_program->name);
        goto done;
    }

    /* PASS 1: pre-resolve textures before opening the compute encoder. */
    if (mglTessMustEndRenderBeforeCompute(mglRenderEncoderOwnerHasCurrent(
            mglTessDispatchRenderEncoderOwner(&areas)))) {
        mglRendererEndRenderEncodingPort(renderer);
    }
    MGLRenderCommandBufferState command_state = {0};
    const int has_command_state = mglRenderCommandBufferOwnerHasState(
        mglTessDispatchCommandBufferOwner(&areas), &command_state);
    if (mglTessCommandBufferNeedsNew(has_command_state, command_state.status)) {
        /* -newCommandBuffer was METAL_LOCK + -newCommandBufferLocked +
         * METAL_UNLOCK; the lock is the GL-thread assertion. */
        MGL_ASSERT_GL_THREAD();
        if (!mglRenderPassNewCommandBufferLocked(renderer)) {
            fprintf(stderr,
                    "MGL TESS ERROR: failed to create command buffer for TES "
                    "compute\n");
            goto done;
        }
    }

    MGLTessTextureBind tes_texture_binds[TEXTURE_UNITS * 2u];
    const uint32_t tes_texture_bind_count = mglTessCollectTextureBinds(
        glm_ctx, tes_program, _TESS_EVALUATION_SHADER, tes_texture_binds,
        (uint32_t)(sizeof(tes_texture_binds) / sizeof(tes_texture_binds[0])));
    if (!mglTessEnsureTextureMetalData(renderer, tes_texture_binds,
                                       tes_texture_bind_count, glm_ctx)) {
        goto done;
    }

    MGLTessStageBufferBindingList stage_buffer_bindings = {0};
    if (!mglTessPrepareStageBufferBindings(renderer, &stage_buffer_bindings,
                                           _TESS_EVALUATION_SHADER,
                                           &stage_copy_backs)) {
        goto done;
    }

    MGLRenderComputeExecutionPlan execution_plan = {0};
    execution_plan.pipeline = tes_pipeline;
    void *patch_inputs = mglRendererBackendGetTcsPatchOutBuffer(areas.backend);
    mglTessDispatchKeepAlive(temporaries, patch_inputs);
    if (!mglTessDispatchPlanBufferOrBind(
            &execution_plan, temporaries, tess_factor_buffer, 0u,
            MGL_AIR_TESS_SLOT_TESS_FACTOR) ||
        !mglTessDispatchPlanBufferOrBind(
            &execution_plan, temporaries,
            patch_inputs ? patch_inputs : out_buffer, 0u,
            MGL_AIR_TESS_SLOT_PATCH_OUT) ||
        !mglTessDispatchPlanBufferOrBind(&execution_plan, temporaries,
                                         out_buffer, 0u,
                                         MGL_AIR_TESS_SLOT_TCS_OUTPUT)) {
        goto done;
    }

    if (!mglTessPlanTextureBinds(renderer, tes_texture_binds,
                                 tes_texture_bind_count, glm_ctx,
                                 &execution_plan, temporaries)) {
        goto done;
    }

    if (!mglTessBindPreparedStageBufferBindings(&stage_buffer_bindings, NULL,
                                                &execution_plan, temporaries)) {
        goto done;
    }
    mglTessBindPointSizeParamsToComputeEncoder(renderer, tes_program,
                                               _TESS_EVALUATION_SHADER,
                                               &execution_plan, temporaries);

    /* Transform-feedback stream (slot 31): the kernel writes complete stage
     * records. The renderer gathers selected varyings into the compact GL XFB
     * layout and copies only the prefix containing complete primitives. */
    TransformFeedback *xfb_state =
        mglTessDispatchState(&areas)->transform_feedback;
    Program *gs_program =
        mglResolveProgramForStageFromState(glm_ctx, _GEOMETRY_SHADER);
    /* A monolithic VS+TCS+TES program resolves to itself for the GS stage
     * even with no GS attached.  Guard on the shader slot (same pattern as
     * mglTessClassifyDraw for tcs/tes) so has_gs / the TES→GS handoff and
     * mglTessPlanEvalAfterCompute only see a real geometry stage. */
    if (gs_program && !gs_program->shader_slots[_GEOMETRY_SHADER]) {
        gs_program = NULL;
    }
    const bool xfb_active = mglTessEvalOwnsXFB(glm_ctx, gs_program);
    void *xfb_temporary = NULL;
    void *xfb_copy_destination = NULL;
    Buffer *xfb_destination = NULL;
    size_t xfb_copy_destination_offset = 0u;
    size_t xfb_compact_stride = 0u;
    size_t xfb_copied_vertices = 0u;
    size_t xfb_written_bytes = 0u;
    int xfb_size_ok = 0;
    if (xfb_active) {
        BufferBaseTarget *xfb_slot =
            &mglTessDispatchState(&areas)
                 ->buffer_base[_TRANSFORM_FEEDBACK_BUFFER]
                 .buffers[0];
        size_t capture_vertices = 0u;
        size_t required_bytes = 0u;
        const size_t xfb_session_offset = (size_t)mglXfbSessionOffsetOr(
            (uint64_t)xfb_state->buffer_write_offsets[0], 0u);
        xfb_compact_stride = mglTESXFBVertexStride(tes_program);
        uint32_t capture_verts_u = 0u;
        uint32_t required_bytes_u = 0u;
        const bool size_ok = mglTessPlanEvalXfbCapture(
                                 items_per_instance, eval_instance_count,
                                 (uint32_t)out_stride,
                                 (uint32_t)xfb_compact_stride,
                                 &capture_verts_u, &required_bytes_u) != 0;
        xfb_size_ok = size_ok ? 1 : 0;
        capture_vertices = capture_verts_u;
        required_bytes = required_bytes_u;
        (void)capture_vertices;

        void *xfb_mtl = NULL;
        size_t visible_bytes = 0u;
        if (xfb_slot->buf) {
            if (mglRenderBufferNeedsCPUUpload(xfb_slot->buf->size,
                                              xfb_slot->buf->data.dirty_bits)) {
                /* Consume CPU initialization before the XFB blit writes the
                 * same backing. Otherwise a later map can upload the stale
                 * shadow over the captured GPU data. */
                if (!mglRendererUpdateDirtyBuffer(renderer, xfb_slot->buf)) {
                    goto done;
                }
            } else if (xfb_slot->buf->size == 0) {
                mglRenderClearEmptyBufferDirty(xfb_slot->buf);
            }
            if (!xfb_slot->buf->data.mtl_data) {
                mglRendererBindMTLBuffer(renderer, xfb_slot->buf);
            }
            xfb_mtl = xfb_slot->buf->data.mtl_data;
            mglTessDispatchKeepAlive(temporaries, xfb_mtl);
            if (xfb_mtl) {
                BufferMap xfb_map = {0};
                xfb_map.buf = xfb_slot->buf;
                xfb_map.offset = xfb_slot->offset;
                xfb_map.size = xfb_slot->size;
                visible_bytes = mglBufferMapVisibleBackingBytes(
                    &xfb_map, (size_t)mglTessDispatchBufferLength(xfb_mtl));
            }
        }

        if (mglTessPlanEvalXFBSlot(xfb_active ? 1 : 0, xfb_size_ok) ==
            MGL_TESS_EVAL_XFB_CAPTURE) {
            const GLuint vertices_per_primitive =
                mglTessVerticesPerPrimitive(tes_program);
            MGLTessXFBDestPlan dest_plan = {0};
            const int dest_plan_ok =
                mglTessPlanXFBDestination(
                    items_per_instance, eval_instance_count,
                    (uint32_t)xfb_compact_stride, vertices_per_primitive,
                    (uint64_t)xfb_session_offset, (int64_t)xfb_slot->offset,
                    (uint64_t)visible_bytes, &dest_plan) &&
                dest_plan.valid;
            const int dest_ok = mglTessEvalXFBDestReady(
                xfb_mtl ? 1 : 0, xfb_slot->buf != NULL, dest_plan_ok);
            /* The AIR kernel writes full stage records (built-ins followed by
             * location-based user outputs). GL XFB is a compact stream of only
             * the selected varyings, so it can never target the GL range
             * directly. Gather the selected fields after the dispatch. */
            xfb_temporary = mglTessDispatchCreateBuffer(
                required_bytes, MGL_TESS_DISPATCH_STORAGE_SHARED);
            mglTessDispatchAdopt(temporaries, xfb_temporary);
            if (!xfb_temporary) {
                goto done;
            }
            if (!mglTessDispatchPlanBufferOrBind(&execution_plan, temporaries,
                                                 xfb_temporary, 0u,
                                                 MGL_AIR_TESS_SLOT_XFB_OUT)) {
                goto done;
            }
            if (dest_ok) {
                xfb_copied_vertices = dest_plan.copied_vertices;
                xfb_written_bytes = dest_plan.written_bytes;
                xfb_copy_destination = xfb_mtl;
                xfb_copy_destination_offset = dest_plan.destination_offset;
                xfb_destination = xfb_slot->buf;
            }
        }
    }
    if (mglTessPlanEvalXFBSlot(xfb_active ? 1 : 0, xfb_size_ok) ==
        MGL_TESS_EVAL_XFB_DUMMY) {
        /* The TES compute kernel always declares and writes the XFB stream
         * slot (31); bind a 1-byte dummy so the slot is never dangling when
         * GL feedback is inactive. */
        const uint64_t dummy_bytes = mglTessDummyXfbBytes((uint64_t)out_size);
        void *cached_dummy = NULL;
        void *xfb_dummy = NULL;
        if (mglRendererBackendGetTessXfbDummyBuffer(areas.backend, dummy_bytes,
                                                    &cached_dummy) == 1) {
            xfb_dummy = cached_dummy;
            mglTessDispatchKeepAlive(temporaries, xfb_dummy);
        }
        if (!xfb_dummy) {
            xfb_dummy = mglTessDispatchCreateBuffer(
                (size_t)dummy_bytes, MGL_TESS_DISPATCH_STORAGE_SHARED);
            mglTessDispatchAdopt(temporaries, xfb_dummy);
            if (xfb_dummy) {
                (void)mglRendererBackendPutTessXfbDummyBuffer(areas.backend,
                                                              xfb_dummy);
            }
        }
        if (xfb_dummy) {
            if (!mglTessDispatchPlanBufferOrBind(&execution_plan, temporaries,
                                                 xfb_dummy, 0u,
                                                 MGL_AIR_TESS_SLOT_XFB_OUT)) {
                goto done;
            }
        }
    }

    const int indexed = areas.tessellation->tessIndexedDraw ? 1 : 0;
    uint32_t gather_verts = 0u;
    uint32_t gather_prims = 0u;
    mglTessPlanEvalGather(indexed,
                          (uint32_t)areas.tessellation->tessInstanceRecords,
                          contract->patch_vertices, patch_count, &gather_verts,
                          &gather_prims);
    MGLTessEvalPerPatchDispatchSpec patch_spec;
    mglTessFillEvalPerPatchSpec(
        gl_in_buffer, (uint64_t)gl_in_offset, (uint64_t)gl_in_instance_stride,
        indexed ? control_point_index_buffer : NULL, gather_verts, gather_prims,
        indexed, (uint32_t)gl_in_vertices, patch_count, eval_instance_count,
        items_per_instance, &patch_spec);
    void *patch_keep_alive = NULL;
    if (!mglTessAppendEvalPerPatchDispatches(&execution_plan, tes_program,
                                             factor_bytes, &patch_spec,
                                             &patch_keep_alive)) {
        free(patch_keep_alive);
        goto done;
    }
    if (patch_keep_alive) {
        /* The method wrapped the block in an NSData with a free() deallocator
         * so the plan's byte pointers stayed valid; kCFAllocatorMalloc is the
         * same contract (the block was malloc'd and is freed with free()). */
        CFDataRef keep = CFDataCreateWithBytesNoCopy(
            kCFAllocatorDefault, (const UInt8 *)patch_keep_alive, 1,
            kCFAllocatorMalloc);
        if (!keep) {
            free(patch_keep_alive);
            goto done;
        }
        mglRendererTemporariesAdd(temporaries, (void *)keep);
        CFRelease(keep); /* the temporaries set holds its own reference */
    }
    {
        MGLRenderCopyBackEntry copy_back_entries[kMGLMaxBufferSlots] = {0};
        uint32_t copy_back_entry_count = mglRenderCollectCopyBackEntries(
            (const MGLRenderCopyBackEntry *)stage_copy_backs.slots,
            kMGLMaxBufferSlots, copy_back_entries, kMGLMaxBufferSlots);
        execution_plan.barrier_scope = MGL_RENDER_COMPUTE_BARRIER_BUFFERS;
        MGLRenderComputeExecutionResult execution_result = {0};
        char execution_error[256] = {0};
        if (mglRenderExecuteComputeExecutionPlan(
                mglTessDispatchCommandBufferOwner(&areas),
                areas.gpu_recovery_command_owner
                    ? *areas.gpu_recovery_command_owner
                    : NULL,
                &execution_plan, copy_back_entries, copy_back_entry_count, 1u,
                &execution_result, execution_error,
                sizeof(execution_error)) != 0) {
            if (execution_result.transaction.device_reset_requested) {
                atomic_store_explicit(&areas.core->deviceResetRequested, true,
                                      memory_order_release);
            }
            fprintf(stderr, "MGL TESS ERROR: C++ TES execution failed: %s",
                    execution_error[0] ? execution_error : "unknown error");
            goto done;
        }
    }

    if (mglTessXFBCopyBackReady((uint64_t)xfb_written_bytes,
                                xfb_temporary ? 1 : 0,
                                xfb_destination ? 1 : 0)) {
        const uint8_t *src_base =
            (const uint8_t *)mglTessDispatchBufferContents(xfb_temporary);
        if (!src_base) {
            fprintf(stderr, "MGL TESS XFB: missing temporary contents\n");
            goto done;
        }
        const bool separate_attribs =
            mglXfbSeparateAttribs(tes_program->transform_feedback_buffer_mode) !=
            0;
        if (separate_attribs) {
            /* One GL buffer binding per varying (GL 4.6 §11.1.3.2). */
            for (GLsizei varying = 0;
                 varying < tes_program->transform_feedback_varying_count;
                 varying++) {
                if (!mglXfbVaryingSlotValid((uint32_t)varying)) {
                    break;
                }
                const char *name =
                    tes_program->transform_feedback_varying_names[varying];
                uint32_t record_offset = 0u;
                uint32_t field_type = 0u;
                uint32_t field_bytes = 0u;
                if (!mglTessResolveXFBSource(tes_program, name, &record_offset,
                                             &field_type, &field_bytes)) {
                    continue;
                }
                (void)record_offset;
                (void)field_type;
                BufferBaseTarget *slot =
                    &mglTessDispatchState(&areas)
                         ->buffer_base[_TRANSFORM_FEEDBACK_BUFFER]
                         .buffers[varying];
                Buffer *dest_buf = slot->buf;
                if (!dest_buf) {
                    continue;
                }
                if (mglRenderBufferNeedsCPUUpload(dest_buf->size,
                                                  dest_buf->data.dirty_bits)) {
                    if (!mglRendererUpdateDirtyBuffer(renderer, dest_buf)) {
                        goto done;
                    }
                }
                if (!dest_buf->data.mtl_data) {
                    mglRendererBindMTLBuffer(renderer, dest_buf);
                }
                void *dest_mtl = dest_buf->data.mtl_data;
                /* SubData below may replace the buffer's Metal backing, and
                 * the ARC local was what kept this one alive (log 128). */
                mglTessDispatchKeepAlive(temporaries, dest_mtl);
                const uint64_t session_offset = mglXfbSessionOffsetOr(
                    (uint64_t)xfb_state->buffer_write_offsets[varying], 0u);
                uint64_t visible = 0u;
                if (dest_mtl && slot->offset >= 0) {
                    BufferMap xfb_map = {0};
                    xfb_map.buf = dest_buf;
                    xfb_map.offset = slot->offset;
                    xfb_map.size = slot->size;
                    visible = (uint64_t)mglBufferMapVisibleBackingBytes(
                        &xfb_map, (size_t)mglTessDispatchBufferLength(dest_mtl));
                }
                MGLXfbVsBufferDest dest = {0};
                if (!mglXfbPlanVsBufferDestOrUnbacked(
                        (uint32_t)xfb_copied_vertices, field_bytes,
                        dest_mtl ? 1 : 0, slot->offset, session_offset, visible,
                        &dest) ||
                    dest.skip) {
                    continue;
                }
                size_t dest_offset = (size_t)dest.destination_offset;
                size_t max_verts = dest.written_records;
                size_t written = dest.written_bytes;
                uint8_t *packed = (uint8_t *)calloc(1u, written);
                if (!packed) {
                    fprintf(stderr,
                            "MGL TESS XFB: OOM packing separate attrib %d",
                            (int)varying);
                    goto done;
                }
                mglTessPackXFBSeparate(tes_program, name, src_base,
                                       (uint32_t)out_stride, (uint32_t)max_verts,
                                       packed);
                mglRendererBufferSubData(glm_ctx, dest_buf,
                                         (GLintptr)dest_offset,
                                         (GLsizeiptr)written, packed);
                if (dest_mtl) {
                    uint8_t *live =
                        (uint8_t *)mglTessDispatchBufferContents(dest_mtl);
                    if (live) {
                        memcpy(live + dest_offset, packed, written);
                    }
                }
                if (mglXfbCPUShadowFits(dest_buf->data.buffer_data ? 1 : 0,
                                        dest_buf->size, (uint64_t)dest_offset,
                                        (uint64_t)written)) {
                    memcpy((uint8_t *)dest_buf->data.buffer_data + dest_offset,
                           packed, written);
                }
                mglRenderMarkBufferCPUWrite(dest_buf, (int64_t)dest_offset,
                                            (int64_t)written);
                free(packed);
            }
        } else {
            uint8_t *packed = (uint8_t *)calloc(1u, xfb_written_bytes);
            if (!packed) {
                fprintf(stderr,
                        "MGL TESS XFB: missing temporary contents or OOM\n");
                goto done;
            }
            mglTessPackXFBInterleaved(tes_program, src_base,
                                      (uint32_t)out_stride,
                                      (uint32_t)xfb_copied_vertices, packed,
                                      (uint32_t)xfb_compact_stride);
            mglRendererBufferSubData(glm_ctx, xfb_destination,
                                     xfb_copy_destination_offset,
                                     xfb_written_bytes, packed);
            /* Mirror into the live Metal allocation: SubData may land in a
             * snapshot while glMapBufferRange serves the CPU shadow. */
            if (xfb_copy_destination) {
                uint8_t *live = (uint8_t *)mglTessDispatchBufferContents(
                    xfb_copy_destination);
                if (live) {
                    memcpy(live + xfb_copy_destination_offset, packed,
                           xfb_written_bytes);
                }
            }
            if (mglXfbCPUShadowFits(xfb_destination->data.buffer_data ? 1 : 0,
                                    xfb_destination->size,
                                    (uint64_t)xfb_copy_destination_offset,
                                    (uint64_t)xfb_written_bytes)) {
                memcpy((uint8_t *)xfb_destination->data.buffer_data +
                           xfb_copy_destination_offset,
                       packed, xfb_written_bytes);
            }
            mglRenderMarkBufferCPUWrite(xfb_destination,
                                        (int64_t)xfb_copy_destination_offset,
                                        (int64_t)xfb_written_bytes);
            free(packed);
        }
    }
    if (mglXfbShouldAdvanceWriteOffset(xfb_active ? 1 : 0,
                                       (uint64_t)xfb_written_bytes)) {
        xfb_state->buffer_write_offsets[0] = mglXfbAdvanceWriteOffset(
            xfb_state->buffer_write_offsets[0], (uint64_t)xfb_written_bytes);
    }

    /* Rasterize through the passthrough vertex stage, or hand the expanded
     * records to a following geometry shader (coverage VS+TC+TE+GS path). */
    const GLenum tess_raster_mode = mglTessRasterGLMode(tes_program);
    MGLTessRasterQueryPlan query = {0};
    mglTessPlanRasterQuery(tes_program, (uint64_t)instance_count,
                           (uint64_t)items_per_instance, xfb_active ? 1 : 0,
                           (uint64_t)xfb_written_bytes,
                           (uint32_t)xfb_compact_stride, &query);
    MGLTessEvalAfterComputePlan after = {0};
    if (!mglTessPlanEvalAfterCompute(
            gs_program ? 1 : 0,
            mglTessDispatchState(&areas)->caps.rasterizer_discard ? 1 : 0,
            items_per_instance, eval_instance_count, &after)) {
        goto done;
    }
    if (after.action == MGL_TESS_AFTER_COMPUTE_GS) {
        if (after.gs_empty) {
            fprintf(stderr,
                    "MGL TESS ERROR: TES→GS empty expansion program=%u",
                    (unsigned)tes_program->name);
            goto done;
        }
        GLsizei gs_count = (GLsizei)after.gs_vertex_count;
        areas.tessellation->pendingGSInputActive = 1;
        areas.tessellation->pendingGSInput =
            (void *)CFRetain((CFTypeRef)out_buffer);
        areas.tessellation->pendingGSInputOffset = 0u;
        areas.tessellation->pendingGSInputStride = out_stride;
        areas.tessellation->pendingGSVertexCount = gs_count;
        /* O1.4: single mglIssue/host path — no ObjC dual call. */
        const int gs_ok = mglDrawHostHandleGeometry(
                              renderer, glm_ctx, tess_raster_mode, 0, gs_count,
                              0, NULL, 0, 1, base_instance,
                              "tessEvalToGeometry")
                              ? 1
                              : 0;
        if (areas.tessellation->pendingGSInput) {
            (void)CFRelease((CFTypeRef)areas.tessellation->pendingGSInput);
            areas.tessellation->pendingGSInput = NULL;
        }
        areas.tessellation->pendingGSInputActive = 0;
        areas.tessellation->pendingGSInputOffset = 0u;
        areas.tessellation->pendingGSInputStride = 0u;
        areas.tessellation->pendingGSVertexCount = 0;
        ok = gs_ok != 0;
        goto done;
    }
    if (after.action == MGL_TESS_AFTER_COMPUTE_DISCARD) {
        /* GL_RASTERIZER_DISCARD: no pixels by definition, so skip the
         * passthrough draw entirely, but the compute expansion already ran
         * and the primitive query must still count the generated
         * primitives (persistent query semantics). */
        areas.batching->currentCommandBufferHasWork = 1;
        mglRecordActivePrimitiveQueryDraw(glm_ctx, query.prims, query.written);
        ok = true;
        goto done;
    }
    if (!mglRenderPassEnsureAIRTessEvalPassthroughFunctionForProgram(renderer, tes_program)) {
        fprintf(stderr,
                "MGL TESS ERROR: TES passthrough vertex unavailable program=%u",
                (unsigned)tes_program->name);
        /* XFB capture already completed above; do not fail the draw and
         * leave transform feedback active when the test only needed feedback. */
        if (mglTessPassthroughFailIsXFBSuccess(xfb_active ? 1 : 0)) {
            mglRecordActivePrimitiveQueryDraw(glm_ctx, query.prims,
                                              query.written);
            ok = true;
            goto done;
        }
        goto done;
    }
    uint32_t prim_type = mglTessRasterPrimitiveType(tes_program);

    areas.tessellation->tessComputeActive = 1;
    areas.tessellation->tessComputeProgram = tes_program;
    const int state_ready = mglRenderPassProcessGLStateLocked(renderer, 1);
    if (!mglTessPassthroughRasterReady(
            state_ready ? 1 : 0,
            mglRenderEncoderOwnerHasCurrent(
                mglTessDispatchRenderEncoderOwner(&areas)),
            mglDrawRasterizationIsEmpty(renderer) ? 1 : 0)) {
        fprintf(stderr,
                "MGL TESS ERROR: TES compute raster skip program=%u "
                "stateReady=%d encoder=%d empty=%d clip0=%d",
                (unsigned)tes_program->name, (int)state_ready,
                mglRenderEncoderOwnerHasCurrent(
                    mglTessDispatchRenderEncoderOwner(&areas)),
                (int)mglDrawRasterizationIsEmpty(renderer),
                areas.ctx && mglTessDispatchState(&areas)->caps.clip_distances[0]
                    ? 1
                    : 0);
        areas.tessellation->tessComputeActive = 0;
        areas.tessellation->tessComputeProgram = NULL;
        if (mglTessPassthroughFailIsXFBSuccess(xfb_active ? 1 : 0)) {
            mglRecordActivePrimitiveQueryDraw(glm_ctx, query.prims,
                                              query.written);
            /* Feedback already landed; returning 0 would raise
             * INVALID_OPERATION and skip the test's EndTransformFeedback. */
            ok = true;
            goto done;
        }
        goto done;
    }

    mglDrawApplyPolygonOffset(renderer, tess_raster_mode);
    for (GLsizei i = 0; i < instance_count; i++) {
        size_t instance_offset = (size_t)mglTessPassthroughInstanceOffset(
            (uint32_t)i, items_per_instance, (uint32_t)out_stride);
        void *encoder_owner = mglTessDispatchRenderEncoderOwner(&areas);
        mglTessDispatchSetRenderVertexBuffer(encoder_owner, out_buffer,
                                             instance_offset, 0u);
        mglTessDispatchDrawPrimitives(encoder_owner, prim_type, 0u,
                                      (size_t)items_per_instance, 1u,
                                      (size_t)base_instance + (size_t)i);
    }
    areas.batching->currentCommandBufferHasWork = 1;
    mglRecordActivePrimitiveQueryDraw(glm_ctx, query.prims, query.written);
    areas.tessellation->tessComputeActive = 0;
    areas.tessellation->tessComputeProgram = NULL;
    ok = true;

done:
    /* The method cleared the list on every path after a prepare attempt; an
     * unregistered key is a no-op in the backend, so one call here is the
     * same.  The temporaries set and the pipeline (+1 each) are released
     * after the plan has been encoded, and the kernel's stage-in keep-alive
     * block lives in the set for exactly as long as the plan does. */
    mglClearStageBindingCopyBacks(renderer, &stage_copy_backs);
    if (temporaries) {
        mglRendererTemporariesRelease(temporaries);
    }
    CFRelease((CFTypeRef)tes_pipeline);
    return ok;
}
