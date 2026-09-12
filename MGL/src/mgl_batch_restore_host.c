/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * mgl_batch_restore_host.c — the renderer-side driver of the restore-from-key
 * plan (mgl_batch_restore.c).  It lives in its own TU because the plan file
 * must stay linkable on its own for test_batch_restore, while this driver
 * calls into the renderer / state layer (program-pipeline restore, hashtable
 * lookups, framebuffer-binding sync).
 *
 * Split out of mgl_batch_restore.c, where it first landed and where
 * `make test-all` immediately caught the link failure -- the same lesson the
 * batch RT-mark host TU was created for.
 */

#include "mgl_batch_restore.h"
#include "mgl_render.h"          /* mglRestoreProgramPipelinePair, sync names */
#include "hash_table.h"          /* searchHashTable */
#include "mgl_types_state.h"     /* mglInvalidateStateHashCachesForDirtyBits */

/* Context for the restore-from-key driver. */
typedef struct MGLKeyRestoreCtx_t {
    GLMContext glm;
} MGLKeyRestoreCtx;

/* === Restore-from-key driver ===
 * Former -[MGLRenderer restoreStateFromKey:context:].  Every callback it needs
 * is plain C (program-pipeline pair restore, hashtable lookups, viewport and
 * scissor writes), so the orchestration needs no port at all. */
static void mglKeyRestoreProgram(void *v, uint32_t program, uint32_t pipeline)
{
    mglRestoreProgramPipelinePair(((MGLKeyRestoreCtx *)v)->glm, program, pipeline);
}

static void mglKeyRestoreVao(void *v, uint32_t name)
{
    GLMContext glm = ((MGLKeyRestoreCtx *)v)->glm;
    if (name == (glm->active_state->vao ? glm->active_state->vao->name : 0)) return;
    glm->active_state->vao = name
        ? (VertexArray *)searchHashTable(&glm->active_state->vao_table, name) : NULL;
}

static void mglKeyRestoreFbo(void *v, uint32_t name)
{
    GLMContext glm = ((MGLKeyRestoreCtx *)v)->glm;
    uint32_t cur = glm->active_state->framebuffer
                       ? glm->active_state->framebuffer->name : 0;
    if (name == cur) return;
    glm->active_state->framebuffer = name
        ? (Framebuffer *)searchHashTable(&glm->active_state->framebuffer_table, name)
        : NULL;
}

static void mglKeyRestoreSyncFboNames(void *v)
{
    mglRendererSyncFramebufferBindingNames(((MGLKeyRestoreCtx *)v)->glm);
}

static void mglKeyRestoreViewportScissor(void *v, const int32_t vp[4], int sc_en,
                                         const int32_t sc[4])
{
    GLMContext glm = ((MGLKeyRestoreCtx *)v)->glm;
    for (int i = 0; i < 4; i++) glm->active_state->viewport[i] = vp[i];
    if (sc_en) {
        glm->active_state->caps.scissor_test = true;
        for (int i = 0; i < 4; i++) glm->active_state->var.scissor_box[i] = sc[i];
    } else {
        glm->active_state->caps.scissor_test = false;
    }
}

void mglBatchRestoreStateFromKey(const MGLStateKey *key, GLMContext glm_ctx)
{
    if (!key || !glm_ctx) return;
    MGLKeyRestoreCtx c = {glm_ctx};
    MGLBatchRestoreFromKeyOps ops = {
        .ctx = &c, .program_name = key->program_name,
        .program_pipeline_name = key->program_pipeline_name,
        .vao_name = key->vao_name, .fbo_name = key->fbo_name,
        .scissor_enabled = key->scissor_enabled,
        .restore_program = mglKeyRestoreProgram, .set_vao = mglKeyRestoreVao,
        .set_fbo = mglKeyRestoreFbo, .sync_fbo_names = mglKeyRestoreSyncFboNames,
        .apply_viewport_scissor = mglKeyRestoreViewportScissor,
    };
    for (int i = 0; i < 4; i++) {
        ops.viewport[i] = key->viewport[i];
        ops.scissor[i] = key->scissor[i];
    }
    mgl_batch_restore_apply_from_key(&ops);
}
