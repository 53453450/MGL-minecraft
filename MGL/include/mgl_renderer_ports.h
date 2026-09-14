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
#include "mgl_tessellation_state.h"   /* MGLTessellationState, MGLGeometryState */
#include "mgl_region_value.h"       /* MGLRegionValue / MGLSizeValue / MGLOriginValue (log 149) */

/* Forward declaration: the blend record lives in mgl_render.h, which this
 * header does not need to pull in. */
struct MGLRenderPipelineBlendState_t;

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

/* Bind one Texture's Metal object (mglRendererBindMTLTexture, mgl_texture_bind.h)
 * is C now: it was -[MGLRenderer bindMTLTextureLocked:], so the former
 * mglRendererBindMTLTexturePort is gone and callers link straight to it. */

/* ---- texture materialization ports --------------------------------------
 * The four Objective-C steps left inside mglRendererBindMTLTexture: Metal
 * texture creation (the -createMTLTextureFromGLTexture: /
 * -createFallbackMTLTexture: pair), the two CPU-data uploads, and the
 * render-target preservation that needs the render-pass manager.  They live in
 * MGLRenderer+Texture.m, so their C entries stay in the shell TU until that
 * file is converted.
 *
 * OWNERSHIP: both creation ports return +1 (the caller owns it, releases it
 * with mglSafeReleaseMetalObj); the upload ports return 1 on success and write
 * *out_all_levels_uploaded when it is non-NULL. */
void *mglRendererCreateMTLTextureFromGLTexturePort(void *renderer, Texture *tex);
void *mglRendererCreateFallbackMTLTexturePort(void *renderer, Texture *tex);
int mglRendererUploadFullCPUTextureDataPort(void *renderer, Texture *tex,
                                            void *texture,
                                            const char *reason);
int mglRendererUploadDirtyCPUTextureDataPort(void *renderer, Texture *tex,
                                             void *texture,
                                             uint32_t pixel_format,
                                             uint32_t num_faces,
                                             uint32_t upload_level_count,
                                             int is_array,
                                             int texture1d_backed_by_2d,
                                             int texture1d_array_backed_by_2d_array,
                                             uint32_t tex_type,
                                             int *out_all_levels_uploaded);

/* ---- dyn-bind / sampler ports ------------------------------------------ */

/* The binding-state owner (the object the dyn-bind plans write bindings
 * through) is reached as `areas.binding_state_owner`, not through a port. */

/* Buffer staging for the dyn-vertex path.  Both halves are C now: the dirty
 * base-buffer list goes to mglRenderUpdateDirtyBaseBufferList(ctx, list, where)
 * and the Metal allocation bind to mglRendererBindMTLBuffer (METAL_LOCK() is
 * only MGL_ASSERT_GL_THREAD(), so there is no lock to keep in Objective-C). */
void mglRendererBindMTLBuffer(void *renderer, Buffer *buffer);

/* Binding-state push for the mapper fallback path. */
/* mglRendererMapBuffersToMTLPort is gone: the buffer mapping is the C function
 * mglRendererMapBuffersToMTL (mgl_buffer_map.h). */
/* mglRendererBindTexturesToCurrentRenderEncoderPort is gone: its target is C now
 * (mglBindTexturesToCurrentRenderEncoder, mgl_sampled_sampler.h). */
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
    /* Pipeline-cache object (the Objective-C cache instance) and the pieces of
     * the tessellation state the vertex-descriptor plan needs.  Both are plain
     * values C can hold, so they travel in the areas instead of costing a port
     * per field; the shell fills them. */
    void *pipeline_cache_object;
    /* ADDRESS of the renderer's command-recovery owner slot (it can be swapped
     * while a recovery runs, so dereference it at the point of use, like
     * binding_state_owner). */
    void **gpu_recovery_command_owner;
    /* Sets one attachment's blend factors through the cache's own setter.
     * Returns 1 on success. */
    int (*pipeline_cache_set_blend)(
        void *pipeline_cache_object, uint32_t index,
        const struct MGLRenderPipelineBlendState_t *blend);
    int32_t tess_native_tes_active;
    void *tess_native_tes_program;
    /* The tessellation and geometry records themselves (the C draw host port
     * reads and writes their fields; the two scalars above are the older
     * single-field shortcut and stay for the callers that only need those). */
    MGLTessellationState *tessellation;
    MGLGeometryState *geometry;
    uint32_t tess_tcs_output_stride;
    uint32_t tess_cull_capture_first_instance;
    uint32_t tess_cull_capture_instance_stride;
} MGLRendererStateAreas;

void mglRendererStateAreasPort(void *renderer, MGLRendererStateAreas *areas_out);

/* Make sure the current command buffer is writable (rotating it when it was
 * already committed).  A port: the rotation runs -newCommandBufferLocked /
 * -endRenderEncodingLocked, which are Objective-C render-pass methods. */
int mglRendererEnsureWritableCommandBufferPort(void *renderer,
                                               const char *reason);

/* ---- compute / tessellation host entries ---------------------------------
 * The C compute and tessellation binders need the renderer's program, encoder
 * and stage copy-back plumbing.  Those bodies still live in MGLRenderer.m and
 * MGLRenderer+RenderPass.m, so they arrive through these entries until those
 * two files are converted; they are thin forwards, no state of their own.
 *
 * OWNERSHIP: mglRendererIsolatedStageBindingBufferPort returns a +1 buffer the
 * caller releases with mglSafeReleaseMetalObj; the sampler entry returns a
 * BORROWED sampler (the renderer or the backend cache owns it). */
/* Assign the renderer's context ivar (the Objective-C compute entry points did
 * `ctx = glm_ctx;` before dispatching; the port-wrapped methods downstream read
 * that ivar through the state areas). */
void mglPlatformShellSetContext(void *renderer, GLMContext glm_ctx);

/* Fresh writing command buffer of the shell (the C entry mgl_ms_sample_loop.c
 * declared locally until now). */
int mglPlatformShellNewCommandBuffer(void *renderer);

/* Fresh writing command buffer of the shell (the C entry mgl_ms_sample_loop.c
 * declared locally until now). */
int mglPlatformShellNewCommandBuffer(void *renderer);

int mglRendererBindMTLProgramPort(void *renderer, Program *program);
void mglRendererEndRenderEncodingPort(void *renderer);
int mglRendererNewCommandBufferLockedPort(void *renderer);
int mglRendererProcessGLStateLockedPort(void *renderer, int draw_command);

/* the Clear*CopyBack port(s) are gone: those methods are C now
 * (mgl_stage_copy_back.h, log 158). */
/* the Clear*CopyBack port(s) are gone: those methods are C now
 * (mgl_stage_copy_back.h, log 158). */
int mglRendererRecordStageBindingCopyBackPort(
    void *renderer, void *copy_backs, uint64_t index, void *temporary,
    void *destination, Buffer *destination_buffer, uint64_t destination_offset,
    uint64_t length);
int mglRendererFlushStageBindingCopyBacksPort(void *renderer, void *copy_backs,
                                              int require_cpu_visibility);
void *mglRendererIsolatedStageBindingBufferPort(void *renderer,
                                                const BufferMap *map,
                                                void *source,
                                                uint64_t required_length);
/* mglRendererMaterializeSampledSamplerPort is gone: the sampler materialize
 * leaf is C now (mglSampledSamplerMaterialize, mgl_sampled_sampler.h) and its
 * callers link straight to it (P0-1, log 151). */

/* === draw / tessellation host entries (phase 2) ==========================
 * The remaining calls mgl_draw_metal_port.m makes into renderer methods, so
 * that file can finish converting.  Thin forwards; retirement follows their
 * targets in MGLRenderer+RenderPass.m / +Tessellation.m / +BindingState.m. */
void mglRendererFlushCommandBufferPort(void *renderer, int finish);
/* Flush any render pass that is currently sampling/drawing into `texture`
 * before it is read back (the method's YES when there was nothing to do).
 * Added for the copyImageSubData leaves (P0-1, log 144). */
int mglRendererSynchronizeRenderPassForTextureReadbackPort(void *renderer,
                                                           void *texture,
                                                           const char *reason);
/* Close a stale render pass when the encoder's FBO no longer matches the
 * current context FBO (the `endRenderPassIfFramebufferChangedForNonDraw:`
 * calls the blit dispatchers make before encoding).  Added for the
 * mtlCopyImageSubData dispatch (P0-1, log 145). */
void mglRendererEndRenderPassIfFramebufferChangedForNonDrawPort(
    void *renderer, uint64_t process_call);
/* Drawable access for the blitFramebuffer attachment resolve (P0-1, log 146).
 * `_drawable` is the `self.drawable` property: next_drawable runs
 * -mglNextDrawable (which assigns the property itself), and drawable_texture
 * is its `.texture` — NULL when there is no drawable, so one NULL check covers
 * both `!_drawable` and `![self mglDrawableTexture]`. */
void mglRendererNextDrawablePort(void *renderer);
void *mglRendererDrawableTexturePort(void *renderer);
int mglRendererEnsureLayerDrawableSizeAtLeastWidthPort(void *renderer,
                                                       size_t required_width,
                                                       size_t required_height,
                                                       const char *reason);
/* copyTexSubImage read-back / upload bridges (P0-1, log 149). */
void mglRendererMTLReadDrawablePort(void *renderer, GLMContext glm_ctx,
                                    void *pixel_bytes, size_t bytes_per_row,
                                    size_t bytes_per_image,
                                    MGLRegionValue region);
/* Sampled-texture readback trace.  The target method takes two NSStrings, so
 * the port takes C strings and the shell makes the NSStrings (P0-1, log 155). */
void mglRendererTraceSampledTextureReadbackPort(
    void *renderer, void *texture, Texture *gl_tex, TextureLevel *level0,
    GLuint program, GLuint binding, const char *stage, const char *reason,
    uint64_t hit);

/* Whether the current render pass references `texture` (used by the sampled
 * render-target copy repair path, P0-1 log 150). */
int mglRendererCurrentRenderPassUsesTexturePort(void *renderer, void *texture);

int mglRendererCopyTextureUploadWithDedicatedCommandBufferPort(
    void *renderer, void *source_buffer, size_t source_offset,
    size_t source_bytes_per_row, size_t source_bytes_per_image,
    size_t source_layer_stride, size_t layer_count, MGLSizeValue source_size,
    void *texture, size_t destination_slice, size_t destination_level,
    MGLOriginValue destination_origin, const char *reason);
int mglRendererEnsureRasterEncoderForDrawPort(void *renderer);
int mglRendererPrepareEmulatedIndirectCPUReadPort(void *renderer,
                                                  GLMContext draw_ctx,
                                                  const char *label);
int mglRendererEnsureAIRGeometryPassthroughPort(void *renderer,
                                                Program *program,
                                                uint32_t output_primitive);
/* The TES-vertex passthrough function is still an Objective-C method in
 * MGLRenderer+RenderPass.m; this port retires with it.  Added by log 128. */
int mglRendererEnsureAIRTessEvalPassthroughPort(void *renderer, Program *program);


/* GPU capture (the MGLPlatformRendererShell class owns the Metal capture
 * session): start reads MGL_GPU_CAPTURE, stop is unconditional. */
void mglPlatformShellGpuCaptureStart(void *renderer);
void mglPlatformShellGpuCaptureStop(void *renderer);

/* Keep-alive set for the temporaries a binding plan borrows (the Objective-C
 * side used an NSMutableArray).  Create returns +1; add retains the object for
 * the set's lifetime; release drops the whole set. */
void *mglRendererTemporariesCreate(void);
void mglRendererTemporariesAdd(void *temporaries, void *object);
void mglRendererTemporariesRelease(void *temporaries);

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
