/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Renderer state areas / read-only view (ARCH-010 / Phase 5).
 * L2 drivers that only need FillStateAreas/View include this instead of
 * the wide mgl_renderer_ports.h umbrella.
 */

#ifndef MGL_RENDERER_STATE_AREAS_H
#define MGL_RENDERER_STATE_AREAS_H

#include "glm_context.h"
#include "mgl_trace_strategy.h"
#include "mgl_batching_state.h"
#include "mgl_renderer_core_state.h"
#include "mgl_command_state.h"
#include "mgl_pipeline_cache_state.h"
#include "mgl_gpu_recovery_state.h"
#include "mgl_tessellation_state.h"

struct MGLRenderPipelineBlendState_t;
struct MGLRenderPipelineDescriptorState;
struct MGLRenderDepthStencilDescriptorState_t;

#ifdef __cplusplus
extern "C" {
#endif

/* === Renderer state areas (the "one struct, one port" pattern) ===
 * States whose records are plain C structs are handed out as pointers, so C
 * drivers read and write their fields directly instead of going through one
 * port per field.  Fetched explicitly (caller-provided storage, no hidden
 * static), and cheap enough to take once per driver call.
 *
 * `binding_state_owner` is the ADDRESS of the owner slot: the value can change
 * while a driver runs, so dereference it at the point of use. */
typedef struct MGLRendererStateAreas {
    MGLRendererCoreState *core;
    /* The backend handle and the owning context: blit/texture code needs both
     * (cache lookups and error dispatch) and they are plain C pointers. */
    void *backend;
    /* The render pass manager object itself (the .m's `_renderPassManager`); the
     * copy-back flush needs it whole for the pass-manager entry points, not just
     * its `state` (log 159). */
    void *render_pass_manager;
    GLMContext ctx;
    MGLBatchingState *batching;
    /* Mutable: the trace-replay identity is written by the flush driver. */
    MGLCommandState *command;
    const MGLPipelineCacheState *pipeline_cache;
    void **binding_state_owner;
    MGLFragmentTextureTraceBinding *fragment_trace_bindings;
    /* ADDRESS of the renderer's command-recovery owner slot (it can be swapped
     * while a recovery runs, so dereference it at the point of use, like
     * binding_state_owner). */
    void **gpu_recovery_command_owner;
    /* The renderer's live GPU-recovery record (the PSO build path writes the
     * interface-mismatch and quarantine fields). */
    MGLGPURecoveryState *gpu_recovery;
    int32_t tess_native_tes_active;
    void *tess_native_tes_program;
    /* The tessellation and geometry records themselves (the C draw host port
     * reads and writes their fields; the two scalars above are the older
     * single-field shortcut and stay for the callers that only need those). */
    MGLTessellationState *tessellation;
    MGLGeometryState *geometry;
    uint32_t tess_tcs_output_stride;
    uint32_t tess_cull_capture_first_instance;
    /* The current drawable (trace logging only) and the query-state owner
     * pointer the render encoder attaches visibility results from. */
    void *drawable;
    void *query_state_owner;
    /* Snapshots of fields that live in Objective-C-only records: the GPU
     * recovery quarantine gate and the emulated-MS forced sample id.  Both are
     * read-only at the point of use. */
    uint32_t gpu_interface_mismatch_blocked_program;
    double gpu_interface_mismatch_blocked_until;
    int32_t mssample_forced_id;
    /* The CAMetalLayer (trace/present checks) and the swap interval: both are
     * stable for the duration of a call, so a snapshot is enough. */
    void *layer;
    int32_t swap_interval;
    uint32_t tess_cull_capture_instance_stride;
} MGLRendererStateAreas;

void mglRendererFillStateAreas(void *renderer, MGLRendererStateAreas *areas_out);

/* === Setter ports ===========================================================
 * The few fields a driver changes in the renderer's state, one narrow function
 * each, so the drivers that only read can hold the read-only view below. */
void mglRendererSetCommandBufferHasWork(void *renderer, int has_work);
void mglRendererRequestDeviceReset(void *renderer);
/* Returns 1 (and clears the request) when a device reset was requested. */
int mglRendererConsumeDeviceReset(void *renderer);
void mglRendererSetDefaultDrawableWritten(void *renderer, int written);
void mglRendererSetLastDrawPrimitiveMode(void *renderer, uint32_t mode);
void mglRendererSetActiveState(void *renderer, GLMState *state);
void mglRendererSetTraceReplayIdentity(void *renderer, uint64_t flush_id,
                                       uint32_t batch_index);
void mglRendererSetDefaultDrawBufferSize(void *renderer, uint32_t draw_buffer,
                                         uint32_t width, uint32_t height);
/* The per-texture-unit fragment trace buffer (trace logging and teardown).
 * Returns the TEXTURE_UNITS entries the trace code fills, or NULL. */
MGLFragmentTextureTraceBinding *mglRendererFragmentTraceBindings(void *renderer);

/* Drop the pipeline-cache entries (device reset / recovery). */
void mglRendererResetPipelineCaches(void *renderer);

/* === Read-only view of the same state ======================================
 * Most drivers only look at the renderer's state.  They take this view, whose
 * records are const and which carries none of the cache bridges, so a write
 * through it does not compile.  A driver that must change state keeps the
 * areas above and does so through the owner it names (render pass state
 * setters, mglMarkRendererDirtyBits, ...).  `binding_state_owner` and
 * `gpu_recovery_command_owner` are slot ADDRESSES like in the areas; read the
 * slot at the point of use. */
/* Pipeline cache as its own port: the cache object, the state it publishes,
 * and the bridges into the cache implementation.  Kept out of the state areas
 * so a reader that only needs the cache does not have to hold the areas (and
 * cannot reach anything else through them). */
typedef struct MGLPipelineCachePort {
    void *object;
    const MGLPipelineCacheState *state;
    int (*set_blend)(void *object, uint32_t index,
                     const struct MGLRenderPipelineBlendState_t *blend);
    int (*blend_state)(void *object, uint32_t index,
                       struct MGLRenderPipelineBlendState_t *blend);
    void (*invalidate)(void *object);
    int (*descriptor_state_for_words)(void *object, const uint64_t *words,
                                      MGLRenderPipelineDescriptorState *state_out);
    int (*create_pso)(void *object,
                      const MGLRenderPipelineDescriptorState *state,
                      void *vertex_function, void *fragment_function,
                      void **pipeline_out, char *error_message,
                      size_t error_capacity);
    void (*store_pipeline)(void *object, void *pipeline, void *vertex_function,
                           void *fragment_function, const uint64_t *words);
    void (*store_descriptor_state)(
        void *object, const MGLRenderPipelineDescriptorState *state,
        const uint64_t *words);
    void (*depth_stencil_state_for_value_state)(
        void *object,
        const struct MGLRenderDepthStencilDescriptorState_t *value_state,
        void **state_out);
    int (*lookup_pipeline)(void *object, const uint64_t *words,
                           void **pipeline_out, void **vertex_function_out,
                           void **fragment_function_out);
    void (*activate)(void *object, void *pipeline, uint32_t color0_format,
                     uint32_t depth_format, uint32_t stencil_format,
                     uint32_t program_name, void *vertex_function,
                     void *fragment_function);
} MGLPipelineCachePort;

/* Fills the port for `renderer`; safe to call per use. */
MGLPipelineCachePort mglRendererPipelineCachePort(void *renderer);

typedef struct MGLRendererStateView {
    const MGLRendererCoreState *core;
    void *backend;
    void *render_pass_manager;
    GLMContext ctx;
    const MGLBatchingState *batching;
    const MGLCommandState *command;
    const MGLPipelineCacheState *pipeline_cache;
    void *const *binding_state_owner;
    void *const *gpu_recovery_command_owner;
    const MGLGPURecoveryState *gpu_recovery;
    const MGLTessellationState *tessellation;
    const MGLGeometryState *geometry;
    int32_t tess_native_tes_active;
    void *tess_native_tes_program;
    uint32_t tess_tcs_output_stride;
    uint32_t tess_cull_capture_first_instance;
    uint32_t tess_cull_capture_instance_stride;
    void *drawable;
    void *query_state_owner;
    uint32_t gpu_interface_mismatch_blocked_program;
    double gpu_interface_mismatch_blocked_until;
    int32_t mssample_forced_id;
    void *layer;
    int32_t swap_interval;
} MGLRendererStateView;

/* Drops the cached renderer state view (mgl_platform_shell.cpp).  The shell
 * setters call this; C hosts changing renderer inputs outside those setters
 * can too. */
void mglRendererStateViewInvalidateCache(void);

static inline void mglRendererFillStateView(void *renderer,
                                            MGLRendererStateView *view_out)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    view_out->core = areas.core;
    view_out->backend = areas.backend;
    view_out->render_pass_manager = areas.render_pass_manager;
    view_out->ctx = areas.ctx;
    view_out->batching = areas.batching;
    view_out->command = areas.command;
    view_out->pipeline_cache = areas.pipeline_cache;
    view_out->binding_state_owner = areas.binding_state_owner;
    view_out->gpu_recovery_command_owner = areas.gpu_recovery_command_owner;
    view_out->gpu_recovery = areas.gpu_recovery;
    view_out->tessellation = areas.tessellation;
    view_out->geometry = areas.geometry;
    view_out->tess_native_tes_active = areas.tess_native_tes_active;
    view_out->tess_native_tes_program = areas.tess_native_tes_program;
    view_out->tess_tcs_output_stride = areas.tess_tcs_output_stride;
    view_out->tess_cull_capture_first_instance =
        areas.tess_cull_capture_first_instance;
    view_out->tess_cull_capture_instance_stride =
        areas.tess_cull_capture_instance_stride;
    view_out->drawable = areas.drawable;
    view_out->query_state_owner = areas.query_state_owner;
    view_out->gpu_interface_mismatch_blocked_program =
        areas.gpu_interface_mismatch_blocked_program;
    view_out->gpu_interface_mismatch_blocked_until =
        areas.gpu_interface_mismatch_blocked_until;
    view_out->mssample_forced_id = areas.mssample_forced_id;
    view_out->layer = areas.layer;
    view_out->swap_interval = areas.swap_interval;
}


#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_STATE_AREAS_H */
