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
#include "mgl_renderer_core_state.h" /* MGLRendererCoreState */
#include "mgl_command_state.h"      /* MGLCommandState */
#include "mgl_pipeline_cache_state.h" /* MGLPipelineCacheState */

#ifdef __cplusplus
extern "C" {
#endif

/* Texture an attachment renders into: the renderbuffer's texture, or the
 * texture object (looked up and cached on first use).  NULL when the
 * attachment carries none -- the C port of
 * -[MGLRenderer framebufferAttachmentTexture:]. */
Texture *mglRendererAttachmentTextureFor(GLMContext ctx, FBOAttachment *att);

/* ---- batch / draw ports ------------------------------------------------
 * Thin wrappers over the renderer entry points the batch replay path drives.
 * They live in one shim TU (MGLPlatformRendererShell.m) so the Objective-C
 * surface C talks to stays in a single place, and each one moves into its
 * implementation file as that file is converted. */

/* MDI argument scratch buffer; *offset_out receives the write offset.
 * C, not a port: the render pass manager's scratch owner lives in the command
 * state (areas.command->mdiArgsScratchOwner) and the allocator itself was
 * already C++ (mglRenderAllocateMDIScratch), so the whole body moved to
 * mgl_renderer_ports.c. */
void *mglRendererMdiScratchBuffer(void *renderer, uint64_t length,
                                  uint64_t *offset_out);

/* The render pass manager's command state (render encoder owner, render-pass
 * state owner, trace identity, FBO-match cache).  C, not a port: it is one
 * field of the state areas. */
const MGLCommandState *mglRendererCommandStateFor(void *renderer);

/* C helper defined in MGLRenderer.m (the Objective-C side keeps its own
 * declaration in MGLRenderer+Draw_Private.h): the batch drivers upload a dirty
 * base-buffer list straight through it. */
bool mglRenderUpdateDirtyBaseBufferList(GLMContext ctx,
                                        BufferMapList *buffer_map_list,
                                        const char *where);

/* Element (index) buffer for a command; fills the GL buffer and, when it has
 * one, the Metal buffer.  0 when the command cannot be resolved.  C, not a
 * port: the bodies only ever needed C helpers, so all three moved here. */
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

/* Renderer state processing (1 = a draw command). */
int mglRendererProcessGLStatePort(void *renderer, int draw_command);

/* Vertex / element / indirect buffer upload for a draw's buffer object: bind
 * the Metal storage when it is missing, then push dirty data.  C, not a port
 * (only the lock-taking bind stays one). */
int mglRendererProcessBuffer(void *renderer, Buffer *buffer);

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
/* The batching state (flags + batch arena) and the command state (including the
 * trace-replay identity) are **not** ports any more: they arrive through
 * MGLRendererStateAreas (`areas.batching` / `areas.command`), which replaced six
 * per-flag wrappers plus a "set the trace identity" wrapper. */

int mglRendererCurrentRenderPassMatchesFramebufferPort(void *renderer);
int mglRendererPrepareRenderPassIfFBOChangedPort(void *renderer, void *batch,
                                                 GLMContext ctx, GLenum *replay_error);

/* Bind one Texture's Metal object through the binding state (returns 0 when
 * the renderer or the texture is missing). */
int mglRendererBindMTLTexturePort(void *renderer, Texture *texture);

/* ---- dyn-bind / sampler ports ------------------------------------------ */

/* The binding-state owner (the object the dyn-bind plans write bindings
 * through) is reached as `areas.binding_state_owner`, not through a port. */

/* Buffer staging for the dyn-vertex path.  Both halves are C now: the dirty
 * base-buffer list goes to mglRenderUpdateDirtyBaseBufferList(ctx, list, where)
 * and the Metal allocation bind to mglRendererBindMTLBuffer (METAL_LOCK() is
 * only MGL_ASSERT_GL_THREAD(), so there is no lock to keep in Objective-C). */
void mglRendererBindMTLBuffer(void *renderer, Buffer *buffer);

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

/* Sampler state for a snapshot key (unretained; the backend cache owns it).
 * C, not a port: the body moved here once the backend handle came from the
 * state areas. */
void *mglRendererSamplerStateForSnapshotKey(void *renderer, const void *key);

/* === Renderer state areas (the "one struct, one port" pattern) ===
 * States whose records are plain C structs are handed out as pointers, so C
 * drivers read and write their fields directly instead of going through one
 * port per field.  Fetched explicitly (caller-provided storage, no hidden
 * static), and cheap enough to take once per driver call.
 *
 * `binding_state_owner` is the ADDRESS of the owner slot: the value can change
 * while a driver runs, so dereference it at the point of use. */
typedef struct MGLRendererStateAreas {
    MGLRendererCoreState *core;
    /* The backend handle and the owning context: blit/texture code needs both
     * (cache lookups and error dispatch) and they are plain C pointers. */
    void *backend;
    GLMContext ctx;
    MGLBatchingState *batching;
    /* Mutable: the trace-replay identity is written by the flush driver. */
    MGLCommandState *command;
    const MGLPipelineCacheState *pipeline_cache;
    void **binding_state_owner;
    MGLFragmentTextureTraceBinding *fragment_trace_bindings;
} MGLRendererStateAreas;

void mglRendererStateAreasPort(void *renderer, MGLRendererStateAreas *areas_out);

/* Make sure the current command buffer is writable (rotating it when it was
 * already committed).  A port: the rotation runs -newCommandBufferLocked /
 * -endRenderEncodingLocked, which are Objective-C render-pass methods. */
int mglRendererEnsureWritableCommandBufferPort(void *renderer,
                                               const char *reason);

/* The pipeline cache's state record (active pipeline handle, pipeline program
 * name, formats).  C readers use the fields directly. */

/* Batch-replay diagnostic trace state: the renderer's fragment texture trace
 * binding records (TEXTURE_UNITS entries, `MGLFragmentTextureTraceBinding`),
 * the pipeline cache's current pipeline state / program name, and the render
 * pass framebuffer name the trace line reports. */

/* Batch-replay trace identity of the current flush / batch. */


#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_PORTS_H */
