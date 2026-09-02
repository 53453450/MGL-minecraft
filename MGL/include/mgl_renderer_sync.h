/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * C/C++ renderer state-sync facade.  GL→Metal synchronization orchestration
 * lives here (or in mgl_renderer_sync.cpp); ObjC retains platform callbacks
 * during migration.
 */

#ifndef MGL_RENDERER_SYNC_H
#define MGL_RENDERER_SYNC_H

#include <stdbool.h>
#include <stdint.h>

#include "glm_context.h"
#include "mgl_frame_activity.h"
#include "mgl_render_pass_coordinator.h"
#include "mgl_sync_domains.h"
#include "mgl_types_framebuffer.h"

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

bool mglRendererObjCProcessGLState(GLMContext context, bool draw_command);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_SYNC_H */
