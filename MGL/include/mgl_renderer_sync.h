/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * C/C++ renderer state-sync facade.  GL→Metal synchronization orchestration
 * lives here (or in mgl_renderer_sync.cpp); ObjC hooks in
 * mgl_renderer_sync_bridge.m and mgl_renderer_sync_ops_bridge.m.
 */

#ifndef MGL_RENDERER_SYNC_H
#define MGL_RENDERER_SYNC_H

#include <stdbool.h>
#include <stdint.h>

#include "glm_context.h"
#include "mgl_frame_activity.h"
#include "mgl_sync_domains.h"
#include "mgl_types_framebuffer.h"

/* Forward declare to avoid cycle:
 * coordinator.h → glm_context.h → backend.h → sync.h → coordinator.h */
typedef struct MGLCommandState_t MGLCommandState;

#ifdef __cplusplus
extern "C" {
#endif

/* Work already performed by processDirtyStateDomains within one
 * processGLState invocation; syncResourceBindings skips these steps. */
typedef struct MGLResourceSyncWork_t {
    bool mappedBuffers;
    bool updatedBaseLists;
    bool boundActiveTextures;
} MGLResourceSyncWork;

/* ObjC/platform hooks invoked by the C++ dirty-domain orchestrator. */
typedef struct MGLRendererSyncOps_t {
    void *renderer;
    bool (*sync_render_pass_for_fbo)(void *renderer, GLMContext context);
    bool (*bind_framebuffer_attachments_in_state_block)(void *renderer,
                                                        GLMContext context);
    bool (*should_defer_buffer_map)(void *renderer, GLMContext context,
                                    int draw_command);
    bool (*map_buffers)(void *renderer, GLMContext context);
    bool (*bind_active_textures)(void *renderer, GLMContext context);
    bool (*update_base_buffer_lists)(void *renderer, GLMContext context);
    bool (*ensure_render_encoder)(void *renderer, GLMContext context,
                                  MGLEncoderCreateReason reason);
    bool (*update_render_encoder)(void *renderer, GLMContext context);
    bool (*sync_pipeline)(void *renderer, GLMContext context,
                          int deferred_buffer_map);
    bool (*sync_incidental_buffer_data)(void *renderer, GLMContext context);
} MGLRendererSyncOps;

/* Platform hooks for FBO/render-pass sync (formerly syncRenderPassStateForContext). */
typedef struct MGLRenderPassSyncOps_t {
    void *renderer;
    Framebuffer *(*get_validated_framebuffer)(void *renderer, GLMContext context,
                                              const char *where);
    bool (*render_pass_matches_framebuffer)(void *renderer, GLMContext context);
    bool (*bind_framebuffer_attachment_textures)(void *renderer,
                                                   GLMContext context);
    bool (*rotate_render_encoder_for_fbo)(void *renderer, GLMContext context);
} MGLRenderPassSyncOps;

int mglRenderProcessGLState(GLMContext context, int draw_command);

int mglRenderProcessDirtyStateDomains(GLMContext context,
                                      uint32_t domain_mask, int draw_command,
                                      const MGLCommandState *command_state,
                                      MGLResourceSyncWork *work,
                                      const MGLRendererSyncOps *ops);

int mglRenderSyncRenderPassForFbo(GLMContext context,
                                  const MGLCommandState *command_state,
                                  const MGLRenderPassSyncOps *ops);

/* Post-dirty-domain processGLState tail (encoder recovery → bind → resource sync). */
typedef struct MGLProcessGLStateTailOps_t {
    void *renderer;
    bool (*recover_nil_render_encoder)(void *renderer, GLMContext context);
    bool (*prepare_draw_pass)(void *renderer, GLMContext context);
    void (*log_draw_pipeline_lookup)(void *renderer, GLMContext context);
    bool (*ensure_pipeline_ready)(void *renderer, GLMContext context,
                                  int trace_process);
    bool (*validate_render_pass)(void *renderer, GLMContext context,
                                 int trace_process);
    bool (*bind_pipeline)(void *renderer, GLMContext context,
                          int trace_process);
    bool (*apply_post_bind_draw_state)(void *renderer, GLMContext context);
} MGLProcessGLStateTailOps;

int mglRenderProcessGLStateTail(
    GLMContext context, const MGLCommandState *command_state,
    int draw_command, int trace_process, MGLResourceSyncWork *resource_sync_work,
    const MGLProcessGLStateTailOps *ops);

/* processGLState preamble (metal recovery → CB ensure) before dirty domains. */
#define MGL_PREAMBLE_FAIL     0
#define MGL_PREAMBLE_CONTINUE 1
#define MGL_PREAMBLE_DONE_OK  2

typedef struct MGLProcessGLStatePreambleOps_t {
    void *renderer;
    bool (*ensure_metal_objects_ready)(void *renderer);
    void (*reject_draw_without_vao)(void *renderer, GLMContext context);
    void (*on_draw_command_begin)(void *renderer, GLMContext context,
                                  MGLCommandState *command_state);
    void (*end_render_pass_non_draw)(void *renderer, uint64_t process_call);
    int (*handle_null_vao_path)(void *renderer, GLMContext context,
                                int draw_command);
    bool (*check_program_quarantine)(void *renderer, GLMContext context);
    bool (*rotate_finalized_command_buffer)(void *renderer, GLMContext context,
                                            int trace_process);
    bool (*create_initial_command_buffer)(void *renderer, GLMContext context,
                                          int trace_process);
} MGLProcessGLStatePreambleOps;

int mglRenderProcessGLStatePreamble(
    GLMContext context, MGLCommandState *command_state, int draw_command,
    uint64_t process_call, int trace_process,
    const MGLProcessGLStatePreambleOps *ops);

bool mglRendererObjCProcessGLState(GLMContext context, bool draw_command);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_SYNC_H */
