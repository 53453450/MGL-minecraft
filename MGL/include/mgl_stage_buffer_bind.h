/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_stage_buffer_bind.h — the per-stage buffer binding drivers moved out of
 * MGLRenderer+BindingState.m (P0-1, log 129).
 *
 *   -bindStageBufferMapEntriesForStage:...  -> mglBindingStateBindStageBufferMapEntries
 *   -bindStageFallbackBuffersForStage:...   -> mglBindingStateBindStageFallbackBuffers
 *
 * Both are called by the two encode drivers in that file
 * (-bindVertexBuffersToCurrentRenderEncoder: / -bindFragmentBuffersToCurrentRenderEncoder:),
 * which stay Objective-C for now and call these entries with the renderer.
 */

#ifndef MGL_STAGE_BUFFER_BIND_H
#define MGL_STAGE_BUFFER_BIND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "glm_context.h"        /* GLMContext, GLuint */
#include "mgl_encode_context.h" /* MGLEncodeContext */
#include "mgl_render.h"         /* MGLRenderBindingSnapshot */
#include "mgl_types_buffer.h"   /* BufferMapList, MAX_BINDABLE_BUFFERS */

#ifdef __cplusplus
extern "C" {
#endif

/* Bind one stage's mapped buffers.  Returns false when the validated buffer
 * lookup or the copy-back recording failed — the same failures the method
 * reported. */
bool mglBindingStateBindStageBufferMapEntries(
    void *renderer, int shader_stage, int is_fragment, BufferMapList *map_list,
    bool *any_binding_present, bool *base_binding_present,
    const bool *attrib_binding_reserved, const MGLEncodeContext *enc_ctx,
    MGLRenderBindingSnapshot *binding_snapshot, uint8_t *byte_scratch,
    size_t *byte_scratch_used, size_t byte_scratch_capacity, int use_snapshot,
    uint32_t max_metal_slots, int allow_isolate_when_gpu,
    int needs_copy_back_on_isolate);

/* Fill every unbound slot of one stage with the backend's fallback buffer. */
void mglBindingStateBindStageFallbackBuffers(
    void *renderer, int shader_stage, int is_fragment, Program *active_program,
    bool *any_binding_present, bool *base_binding_present,
    const MGLEncodeContext *enc_ctx, MGLRenderBindingSnapshot *binding_snapshot,
    int use_snapshot, uint32_t max_metal_slots, int enable_all_slot_fill);

#ifdef __cplusplus
}
#endif

#endif /* MGL_STAGE_BUFFER_BIND_H */
