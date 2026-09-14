/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_storage_image_bind.h — the storage-image binding driver moved out of
 * MGLRenderer+BindingState.m (P0-1, log 130).
 *
 *   -bindStorageImagesForStage:program:bindStage:  -> mglBindingStateBindStorageImagesForStage
 *   -bindStorageImagesForVertexProgram:fragmentProgram: -> mglBindingStateBindStorageImagesForVertexProgram
 *
 * The second one replaces the shell port mglRendererBindStorageImagesForVertexProgramPort
 * (retired in the same cut), so the C draw host calls this entry directly.
 */

#ifndef MGL_STORAGE_IMAGE_BIND_H
#define MGL_STORAGE_IMAGE_BIND_H

#include <stdbool.h>
#include <stdint.h>

#include "glm_context.h"        /* GLMContext */
#include "mgl_types_program.h"  /* Program */

#ifdef __cplusplus
extern "C" {
#endif

/* Bind (or ensure) one stage's storage images.  Returns false on the same
 * failures the method reported. */
bool mglBindingStateBindStorageImagesForStage(void *renderer, int shader_stage,
                                              Program *program,
                                              uint32_t metal_bind_stage);

/* Bind the storage images of the vertex pair (vertex/TES + fragment). */
bool mglBindingStateBindStorageImagesForVertexProgram(void *renderer,
                                                      Program *vertex_program,
                                                      Program *fragment_program);

#ifdef __cplusplus
}
#endif

#endif /* MGL_STORAGE_IMAGE_BIND_H */
