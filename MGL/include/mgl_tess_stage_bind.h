/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_tess_stage_bind.h — the tessellation stage-binding plan and the texture
 * binding plan moved out of MGLRenderer+Tessellation.m (P0-1, log 125).
 *
 * The two records were private typedefs inside the @implementation; they are
 * plain C structs, so they move here as-is with `id __strong` becoming the
 * opaque handle (`void *`) and BOOL becoming int.  The Objective-C methods
 * become C functions taking the renderer handle; the identity of every buffer
 * and every Metal object is unchanged, so the copy-back and temporaries
 * ownership rules stay exactly as they were:
 *
 *   - a buffer created here (+1) is stored straight into the record, which
 *     owns it from then on (the ARC `__strong` field did the same);
 *   - a Metal object handed to the temporaries set is retained by the set,
 *     so a freshly created one drops the creation reference afterwards.
 */

#ifndef MGL_TESS_STAGE_BIND_H
#define MGL_TESS_STAGE_BIND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "glm_context.h"         /* GLMContext, GLuint */
#include "mgl_binding_stage.h"   /* MGLStageBindingCopyBackList */
#include "mgl_draw_tess.h"       /* MGLTessTextureBind */
#include "mgl_render.h"          /* MGLRenderComputeExecutionPlan */
#include "mgl_size_constants.h"  /* kMGLMaxBufferSlots */
#include "mgl_types_buffer.h"    /* Buffer, BufferMapList */

#ifdef __cplusplus
extern "C" {
#endif

/* One Metal buffer bound for a tessellation stage, with the source range of
 * the ordered GPU copy that initializes an isolated binding. */
typedef struct MGLTessStageBufferBinding_t {
    void *buffer;
    size_t offset;
    void *initialization_source;
    size_t initialization_source_offset;
    size_t initialization_length;
    int valid;
} MGLTessStageBufferBinding;

typedef struct MGLTessStageBufferBindingList_t {
    MGLTessStageBufferBinding slots[kMGLMaxBufferSlots];
    void *size_buffer;
    GLuint size_buffer_index;
} MGLTessStageBufferBindingList;

/* Was -prepareTessStageBufferBindings:stage:copyBacks:. */
bool mglTessPrepareStageBufferBindings(void *renderer,
                                       MGLTessStageBufferBindingList *bindings,
                                       int stage,
                                       MGLStageBindingCopyBackList *copy_backs);

/* Was -flushTessStageBindingInitializationBlit:. */
bool mglTessFlushStageBindingInitializationBlit(
    void *renderer, MGLTessStageBufferBindingList *bindings);

/* Was -bindTessStageBufferBindingsToRenderEncoderOwner:bindings:. */
bool mglTessBindStageBufferBindingsToRenderEncoderOwner(
    void *render_encoder_owner, const MGLTessStageBufferBindingList *bindings);

/* Was -bindPreparedTessStageBufferBindings:toComputeEncoder:executionPlan:
 * temporaries:.  The compute encoder argument was already unused. */
bool mglTessBindPreparedStageBufferBindings(
    const MGLTessStageBufferBindingList *bindings, void *compute_command_encoder,
    MGLRenderComputeExecutionPlan *execution_plan, void *temporaries);

/* Was -planTessTextureBinds:count:ctx:plan:temporaries:. */
bool mglTessPlanTextureBinds(void *renderer, const MGLTessTextureBind *binds,
                             uint32_t count, GLMContext ctx,
                             MGLRenderComputeExecutionPlan *plan,
                             void *temporaries);

#ifdef __cplusplus
}
#endif

#endif /* MGL_TESS_STAGE_BIND_H */
