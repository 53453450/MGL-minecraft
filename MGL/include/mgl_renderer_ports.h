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
#include "mgl_trace_strategy.h"    /* MGLFragmentTextureTraceBinding */
#include "mgl_batching_state.h"     /* MGLBatchingState */

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

/* Vertex / element / indirect buffer upload for a draw's buffer object. */
int mglRendererProcessBufferPort(void *renderer, void *buffer);

/* Create an indirect command buffer (indexed when `indexed`), returned with a
 * +1 reference the caller owns and releases.  *failed_out is 1 when Metal
 * raised; the caller then traces its own fallback reason. */
void *mglRendererCreateIndirectCommandBufferPort(void *renderer, int indexed,
                                                 uint64_t count,
                                                 int *failed_out);

/* The @try/@finally frame around one flush (shim): it must tear the replay
 * workspace down even when a draw raises, which C cannot express. */
void mglRendererFlushDrawBufferLockedPort(void *renderer, GLMContext ctx);

/* ---- batch flush / replay-workspace ports ---------------------------------
 * Renderer state the C flush driver reads or writes: the replay-workspace
 * switch and its dual-proxy checkpoint, the batching switches, the trace-replay
 * identity and the render-pass checks. */
/* Binding-state snapshot validity of the renderer's owner. */
int mglRendererBindingStateIsValidPort(void *renderer);

/* The renderer's batching state (flags + batch arena).  C drivers read and
 * write the fields directly; this one port replaced six per-flag wrappers. */
MGLBatchingState *mglRendererBatchingStatePort(void *renderer);

void mglRendererAssertDualProxyPort(void *renderer, GLMContext ctx);
void mglRendererActivateReplayStatePort(void *renderer, GLMContext ctx);
void mglRendererRestoreLiveActiveStatePort(void *renderer, GLMContext ctx);
void mglRendererSetActiveStatePort(void *renderer, GLMContext ctx);
void mglRendererSetCurrentCBHasWorkPort(void *renderer, int has_work);
void mglRendererTraceReplaySetPort(void *renderer, uint64_t flush_id,
                                   uint32_t batch_index);
int mglRendererCurrentRenderPassMatchesFramebufferPort(void *renderer);
int mglRendererPrepareRenderPassIfFBOChangedPort(void *renderer, void *batch,
                                                 GLMContext ctx, GLenum *replay_error);

/* Bind one Texture's Metal object through the binding state (returns 0 when
 * the renderer or the texture is missing). */
int mglRendererBindMTLTexturePort(void *renderer, Texture *texture);

/* ---- dyn-bind / sampler ports ------------------------------------------ */

/* Binding state owner (the object the dyn-bind plans write bindings through). */
void *mglRendererBindingStateOwnerPort(void *renderer);

/* Buffer staging for the dyn-vertex path: upload a dirty base-buffer list and
 * make sure a buffer object has its Metal allocation. */
int mglRendererUpdateDirtyBaseBufferListPort(void *renderer, void *upload);
void mglRendererBindMTLBufferPort(void *renderer, void *buffer);

/* Binding-state push for the mapper fallback path. */
int mglRendererMapBuffersToMTLPort(void *renderer);
int mglRendererBindVertexBuffersToCurrentRenderEncoderPort(void *renderer,
                                                           const void *encode_context);
int mglRendererBindFragmentBuffersToCurrentRenderEncoderPort(void *renderer,
                                                             const void *encode_context);
int mglRendererBindTexturesToCurrentRenderEncoderPort(void *renderer,
                                                      const void *encode_context);
int mglRendererRestoreRenderEncoderAfterTextureUploadPort(void *renderer,
                                                          const char *label);

/* Sampled-resource lookup for the dyn-texture plan. */
uint32_t mglRendererTextureUnitForSampledResourcePort(void *renderer,
                                                      void *resource,
                                                      uint32_t metal_slot,
                                                      int stage);
void *mglRendererTextureForSampledResourcePort(void *renderer, void *resource,
                                               uint32_t metal_slot, int stage,
                                               uint32_t expected_type);

/* Sampler state for a snapshot key (unretained; the backend cache owns it) and
 * the fallback sampler. */
void *mglRendererSamplerStateForSnapshotKeyPort(void *renderer, const void *key);
void *mglRendererFallbackSamplerStatePort(void *renderer);

/* Batch-replay diagnostic trace state: the renderer's fragment texture trace
 * binding records (TEXTURE_UNITS entries, `MGLFragmentTextureTraceBinding`),
 * the pipeline cache's current pipeline state / program name, and the render
 * pass framebuffer name the trace line reports. */
MGLFragmentTextureTraceBinding *mglRendererFragmentTraceBindingsPort(void *renderer);
void *mglRendererPipelineStatePort(void *renderer);
uint32_t mglRendererPipelineProgramNamePort(void *renderer);
uint32_t mglRendererRenderPassFramebufferNamePort(void *renderer);

/* Batch-replay trace identity of the current flush / batch. */
uint64_t mglRendererBatchTraceFlushIdPort(void *renderer);
uint32_t mglRendererBatchTraceBatchIndexPort(void *renderer);

/* Render encoder owner the renderer currently encodes into (or NULL). */
void *mglRendererCurrentRenderEncoderOwnerPort(void *renderer);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_PORTS_H */
