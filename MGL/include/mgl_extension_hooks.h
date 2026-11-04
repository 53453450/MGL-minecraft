/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Extension registration hooks (ARCH-014/025 / Phase 6).
 *
 * Extensions must not include mgl_renderer_ports.h / Private / glsl parser.
 * Hazard helpers: use draw_command.h (mglFlushPendingDrawsForBuffer*,
 * mglPendingDrawsReadBufferRange). Frame arena: mglFrameArenaAlloc.
 */

#ifndef MGL_EXTENSION_HOOKS_H
#define MGL_EXTENSION_HOOKS_H

#include "glm_context_fwd.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct MGLDrawPlan;
struct MGLEncodePlan;
struct MGLEncodeContext;
struct Buffer;

typedef struct MGLExtensionStateVTable {
    void (*init)(GLMContext ctx);
    void (*destroy)(GLMContext ctx);
    /* Segment enter/exit: return dirty mask; must not DIRTY_ALL. */
    uint32_t (*enter_segment)(GLMContext ctx);
    uint32_t (*exit_segment)(GLMContext ctx);
} MGLExtensionStateVTable;

typedef struct MGLGeometryPipelineStage {
    const char *name;
    int (*validate)(GLMContext ctx, const struct MGLDrawPlan *in,
                    struct MGLDrawPlan *out);
    int (*plan)(GLMContext ctx, const struct MGLDrawPlan *in,
                struct MGLEncodePlan *out);
    int (*encode)(struct MGLEncodeContext *enc,
                  const struct MGLEncodePlan *plan);
} MGLGeometryPipelineStage;

void mglRegisterExtensionState(GLMContext ctx, uint32_t ext_id,
                               const MGLExtensionStateVTable *vt);
void mglRegisterGeometryStage(GLMContext ctx,
                              const MGLGeometryPipelineStage *st);

/* Iterate registered geometry stages (for draw/validate drivers). */
size_t mglGeometryStageCount(void);
const MGLGeometryPipelineStage *mglGeometryStageAt(size_t index);

/* True when pending draws read any range of buf (precise hazard query). */
int mglPendingDrawsReferenceBuffer(GLMContext ctx, struct Buffer *buf);

/* Frame arena (N01); NULL if arena disabled. */
void *mglFrameArenaAlloc(GLMContext ctx, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* MGL_EXTENSION_HOOKS_H */
