/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_stage_copy_back.h — the stage binding copy-back list helpers moved out of
 * MGLRenderer.m (P0-1, log 158).
 *
 *   -clearStageBindingCopyBacks:            -> mglClearStageBindingCopyBacks
 *   -clearStageBindingCopyBack:atIndex:     -> mglClearStageBindingCopyBackAtIndex
 *
 * Both retire a port: mglRendererClearStageBindingCopyBacksPort and
 * mglRendererClearStageBindingCopyBackPort are gone and their C callers link
 * straight here.
 */

#ifndef MGL_STAGE_COPY_BACK_H
#define MGL_STAGE_COPY_BACK_H

#include <stddef.h>
#include <stdint.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "glm_context.h"       /* GLMContext (mgl_types_buffer.h needs it) */
#include "mgl_binding_stage.h" /* MGLStageBindingCopyBackList */
#include "mgl_types_buffer.h"  /* Buffer */

#ifdef __cplusplus
extern "C" {
#endif

/* Record one copy-back slot and flush the whole list (P0-1, log 159). */
bool mglRecordStageBindingCopyBack(void *renderer,
                                   MGLStageBindingCopyBackList *copy_backs,
                                   size_t index, void *temporary,
                                   void *destination,
                                   Buffer *destination_buffer,
                                   size_t destination_offset, size_t length);
bool mglFlushStageBindingCopyBacks(void *renderer,
                                   MGLStageBindingCopyBackList *copy_backs,
                                   int require_cpu_visibility);

void mglClearStageBindingCopyBacks(void *renderer,
                                   MGLStageBindingCopyBackList *copy_backs);
void mglClearStageBindingCopyBackAtIndex(void *renderer,
                                         MGLStageBindingCopyBackList *copy_backs,
                                         size_t index);

#ifdef __cplusplus
}
#endif

#endif /* MGL_STAGE_COPY_BACK_H */
