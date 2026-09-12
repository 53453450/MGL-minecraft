/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_renderer_ports.h — C-callable renderer ports (ObjC-zeroing T4).
 *
 * The renderer owns state that C code needs to read: the texture behind an FBO
 * attachment and the render-pass state owner.  Both used to be reachable only
 * through Objective-C messages; these ports keep the accessors in one place so
 * C modules stop needing a category to be linked next to them.
 *
 * The renderer handle is a `void *` because C has no MGLRenderer type; the
 * implementation casts it back.
 */

#ifndef MGL_RENDERER_PORTS_H
#define MGL_RENDERER_PORTS_H

#include "glm_context.h"           /* GLMContext */
#include "mgl_types_framebuffer.h" /* FBOAttachment */
#include "mgl_types_texture.h"     /* Texture */
#include "mgl_types_buffer.h"      /* Buffer */

#ifdef __cplusplus
extern "C" {
#endif

/* Texture an attachment renders into: the renderbuffer's texture, or the
 * texture object (looked up and cached on first use).  NULL when the
 * attachment carries none -- the C port of
 * -[MGLRenderer framebufferAttachmentTexture:]. */
Texture *mglRendererAttachmentTextureFor(GLMContext ctx, FBOAttachment *att);

/* Render pass state owner of a renderer handle, or NULL when there is none. */
void *mglRendererRenderPassStateOwnerPort(void *renderer);

/* ---- batch / draw ports ------------------------------------------------
 * Thin wrappers over the renderer entry points the batch replay path drives.
 * They live in one shim TU (mgl_renderer_port_shim.m) so the Objective-C
 * surface C talks to stays in a single place, and each one moves into its
 * implementation file as that file is converted. */

/* MDI argument scratch buffer; *offset_out receives the write offset. */
void *mglRendererMdiScratchBufferPort(void *renderer, uint64_t length,
                                      uint64_t *offset_out);

/* Element (index) buffer for a command; fills the GL buffer and, when it has
 * one, the Metal buffer.  0 when the command cannot be resolved. */
int mglRendererResolveElementBufferPort(void *renderer, const void *command,
                                        const char *label, GLMContext ctx,
                                        Buffer **gl_buffer_out,
                                        void **mtl_buffer_out);

/* Batch replay tracing for one command. */
void mglRendererTraceReplayCommandPort(void *renderer, void *batch,
                                       void *command, GLMContext ctx,
                                       uint64_t flush_id, uint32_t batch_index,
                                       uint32_t command_index,
                                       const char *phase, const char *reason);

/* Simple-replay fast path for a whole batch. */
int mglRendererTryReplaySimpleBatchPort(void *renderer, void *batch,
                                        GLMContext ctx,
                                        const void *encode_context);

/* Per-command dynamic bindings / sampler snapshot application. */
int mglRendererApplyDynamicBindingsPort(void *renderer, const void *command,
                                        GLMContext ctx, void *encode_context);
int mglRendererApplySamplerSnapshotPort(void *renderer, const void *command,
                                        GLMContext ctx,
                                        const void *encode_context);

/* Cull-distance capture for a direct draw. */
int mglRendererCaptureCullArrayPort(void *renderer, GLMContext ctx, int32_t first,
                                    int32_t count, int32_t instance_count,
                                    uint32_t base_instance);
int mglRendererCaptureCullElementPort(void *renderer, GLMContext ctx,
                                      const uint8_t *index_bytes,
                                      uint32_t index_type, int32_t count,
                                      int32_t base_vertex,
                                      int32_t instance_count,
                                      uint32_t base_instance);

/* Renderer state processing (1 = a draw command). */
int mglRendererProcessGLStatePort(void *renderer, int draw_command);

/* Batch-replay trace identity of the current flush / batch. */
uint64_t mglRendererBatchTraceFlushIdPort(void *renderer);
uint32_t mglRendererBatchTraceBatchIndexPort(void *renderer);

/* Render encoder owner the renderer currently encodes into (or NULL). */
void *mglRendererCurrentRenderEncoderOwnerPort(void *renderer);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_PORTS_H */
