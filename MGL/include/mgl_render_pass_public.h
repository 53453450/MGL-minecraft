/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_render_pass_public.h — Batch / Flush 可见的 Pass 窄面（ARCH-004 / Phase 1）。
 *
 * 目标：mgl_batch_* 不得 include mgl_render_pass_manager_ops.h 内部头；
 * 只经本头调用 Prepare / ProcessGLState / Restore 端口。实现仍转调现有 ops。
 */

#ifndef MGL_RENDER_PASS_PUBLIC_H
#define MGL_RENDER_PASS_PUBLIC_H

#include "glm_context.h"
#include "mgl_state_apply.h" /* mglStateApplyDirtyDomains / MGLResourceSyncWork */

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dirty-domain apply inside processGLState (Flush 驱动). */
bool mglRenderPassProcessDirtyStateDomains(void *renderer, int draw_command,
                                           MGLResourceSyncWork *work);

/* Full GL→Metal state sync before encode. */
int mglRenderPassProcessGLStateLocked(void *renderer, int draw_command);

/* After texture upload closed the encoder. */
int mglRenderPassRestoreRenderEncoderAfterTextureUpload(void *renderer,
                                                        const char *reason);

/* FBO / pass identity prepare (alias of ensure-match for draw). */
int mglRenderPassEnsureCurrentRenderPassMatchesFramebufferForDraw(
    void *renderer);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDER_PASS_PUBLIC_H */
