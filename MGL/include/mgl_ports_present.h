/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Present / drawable / shell capture ports (ARCH-010 / Phase 5).
 */

#ifndef MGL_PORTS_PRESENT_H
#define MGL_PORTS_PRESENT_H

#include "glm_context.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGLRendererLayerMetricsValue_t {
    double drawable_width;
    double drawable_height;
    double frame_width;
    double frame_height;
} MGLRendererLayerMetricsValue;

int mglRendererLayerMetrics(void *renderer,
                            MGLRendererLayerMetricsValue *metrics_out);
void mglPlatformShellSetContext(void *renderer, GLMContext glm_ctx);
int mglPlatformShellNewCommandBuffer(void *renderer);
void mglRendererNextDrawable(void *renderer);
void mglRendererPrefetchNextDrawable(void *renderer);
void *mglRendererDrawableTexture(void *renderer);
int mglRendererEnsureLayerDrawableSizeAtLeastWidth(void *renderer,
                                                   size_t required_width,
                                                   size_t required_height,
                                                   const char *reason);
void mglPlatformShellGpuCaptureStart(void *renderer);
void mglPlatformShellGpuCaptureStop(void *renderer);
void *mglRendererTemporariesCreate(void);
void mglRendererTemporariesAdd(void *temporaries, void *object);
void mglRendererTemporariesRelease(void *temporaries);
void mglRendererTemporariesReleaseOnCommandBuffer(void *temporaries,
                                                  void *command_buffer);

#ifdef __cplusplus
}
#endif

#endif /* MGL_PORTS_PRESENT_H */
