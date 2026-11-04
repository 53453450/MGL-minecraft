/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Batch flush / buffer-process ports (ARCH-010 / Phase 5).
 * Split from mgl_renderer_ports.h so flush drivers avoid the full umbrella.
 */

#ifndef MGL_PORTS_BATCH_FLUSH_H
#define MGL_PORTS_BATCH_FLUSH_H

#include "glm_context.h"
#include "mgl_types_buffer.h" /* Buffer, BufferMapList */
#include "mgl_command_state.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void *mglRendererMdiScratchBuffer(void *renderer, uint64_t length,
                                  uint64_t *offset_out);
const MGLCommandState *mglRendererCommandStateFor(void *renderer);
bool mglRenderUpdateDirtyBaseBufferList(GLMContext ctx,
                                        BufferMapList *buffer_map_list,
                                        const char *where);
int mglRendererResolveElementBufferForDraw(void *renderer, const char *label,
                                           GLMContext ctx, Buffer **gl_out,
                                           void **mtl_out);
int mglRendererResolveElementBufferForCommand(void *renderer,
                                              const void *command,
                                              const char *label, GLMContext ctx,
                                              Buffer **gl_out, void **mtl_out);
int mglRendererResolveElementBuffer(void *renderer, Buffer *gl_element_buffer,
                                    const char *label, GLMContext ctx,
                                    Buffer **gl_out, void **mtl_out);
int mglRendererProcessBuffer(void *renderer, Buffer *buffer);
void *mglRendererCreateIndirectCommandBuffer(void *renderer, int indexed,
                                             uint64_t count, int *failed_out);
void mglRendererFlushDrawBufferLocked(void *renderer, GLMContext ctx);
void mglRendererBindMTLBuffer(void *renderer, Buffer *buffer);
void *mglRendererSamplerStateForSnapshotKey(void *renderer, const void *key);

#ifdef __cplusplus
}
#endif

#endif /* MGL_PORTS_BATCH_FLUSH_H */
