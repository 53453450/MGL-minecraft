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

/* Dirty-domain masks for incremental sync (mirror processDirtyStateDomains). */
#define MGL_SYNC_DOMAIN_FBO          (1u << 0)
#define MGL_SYNC_DOMAIN_STATE        (1u << 1)
#define MGL_SYNC_DOMAIN_PROGRAM_VAO  (1u << 2)
#define MGL_SYNC_DOMAIN_TEX          (1u << 3)
#define MGL_SYNC_DOMAIN_VAO          (1u << 4)
#define MGL_SYNC_DOMAIN_BUFFER       (1u << 5)
#define MGL_SYNC_DOMAIN_RENDER_STATE (1u << 6)
#define MGL_SYNC_DOMAIN_PIPELINE     (1u << 7)
#define MGL_SYNC_DOMAIN_ALL          0xFFFFFFFFu

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

uint32_t mglRenderClassifyDirtySyncDomains(uint32_t dirty_bits);

int mglRenderProcessGLState(GLMContext context, int draw_command);

/* Orchestrates dirty-bit domain dispatch (formerly processDirtyStateDomains). */
int mglRenderProcessDirtyStateDomains(GLMContext context,
                                      uint32_t domain_mask, int draw_command,
                                      const MGLCommandState *command_state,
                                      MGLResourceSyncWork *work,
                                      const MGLRendererSyncOps *ops);

/* ObjC renderer bridge — processGLState body until fully migrated. */
bool mglRendererObjCProcessGLState(GLMContext context, bool draw_command);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_SYNC_H */
