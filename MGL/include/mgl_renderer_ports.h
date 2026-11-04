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
#include "mgl_region_value.h"       /* MGLRegionValue / MGLSizeValue / MGLOriginValue (log 149) */
#include "mgl_renderer_attachment.h" /* AttachmentTextureFor (narrow) */
#include "mgl_ports_batch_flush.h"   /* flush / buffer / MDI (narrow) */
#include "mgl_ports_present.h"       /* drawable / shell / temporaries (narrow) */

/* Forward declaration: the blend record lives in mgl_render.h, which this
 * header does not need to pull in. */
struct MGLRenderPipelineBlendState_t;
/* The value-state descriptor lives in mgl_render.h; the bridges below only pass
 * its address, so a forward declaration is enough here. */
struct MGLRenderPipelineDescriptorState;

#ifdef __cplusplus
extern "C" {
#endif

/* ---- batch / draw ports ------------------------------------------------
 * Declarations live in mgl_ports_batch_flush.h (Phase 5 split). */

/* ---- batch flush / replay-workspace ports ---------------------------------
 * Renderer state the C flush driver reads or writes: the replay-workspace
 * switch and its dual-proxy checkpoint, the batching switches, the trace-replay
 * identity and the render-pass checks. */
/* The batching state (flags + batch arena) and the command state (including the
 * trace-replay identity) are **not** ports any more: they arrive through
 * MGLRendererStateAreas (`areas.batching` / `areas.command`), which replaced six
 * per-flag wrappers plus a "set the trace identity" wrapper. */

/* mglRendererCurrentRenderPassMatchesFramebufferPort is gone (log 193;
 * the header carried this misspelled duplicate of the name retired in
 * log 191): the predicate is C now (mglRenderPassMatchesCurrentFramebuffer). */
/* mglRendererPrepareRenderPassIfFBOChangedPort is gone (log 193): the method is C
 * now (mglRenderPassPrepareIfFBOChanged). */

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
/* mglRendererCreateMTLTextureFromGLTexturePort is gone (log 197):
 * -createMTLTextureFromGLTexture: is C now (mglTextureCreateFromGLTexture). */
/* mglRendererCreateFallbackMTLTexturePort is gone (log 199):
 * -createFallbackMTLTexture: is C now (mglTextureCreateFallback). */
/* mglRendererUploadFullCPUTextureDataPort is gone (log 195):
 * -uploadFullCPUTextureDataIntoTexture:metal:reason: is C now
 * (mglTextureUploadFullCPUData). */
/* mglRendererUploadDirtyCPUTextureDataPort is gone (log 196):
 * -uploadDirtyCPUTextureData:… is C now (mglTextureUploadDirty). */

/* ---- dyn-bind / sampler ports ------------------------------------------ */

/* The binding-state owner (the object the dyn-bind plans write bindings
 * through) is reached as `areas.binding_state_owner`, not through a port. */

/* Buffer / sampler / MDI: mgl_ports_batch_flush.h.
 * Present / shell / temporaries: mgl_ports_present.h. */

/* Binding-state push for the mapper fallback path. */
/* mglRendererMapBuffersToMTLPort is gone: the buffer mapping is the C function
 * mglRendererMapBuffersToMTL (mgl_buffer_map.h). */
/* mglRendererBindTexturesToCurrentRenderEncoderPort is gone: its target is C now
 * (mglBindTexturesToCurrentRenderEncoder, mgl_sampled_sampler.h). */
/* mglRendererRestoreRenderEncoderAfterTextureUploadPort is gone (log 193):
 * -restoreRenderEncoderAfterTextureUploadForDraw: is C now
 * (mglRenderPassRestoreRenderEncoderAfterTextureUpload). */

#include "mgl_renderer_state_areas.h"

/* StateAreas / StateView / PipelineCachePort / setters: see mgl_renderer_state_areas.h (Phase 5). */

/* mglRendererEnsureWritableCommandBufferPort is gone (log 191). */
/* mglRendererBindMTLProgramPort / EndRenderEncodingPort gone (log 193/191). */
/* stage copy-back / isolated buffer / sampled sampler ports gone (log 158–201). */
/* flush / readback / drawable sync ports gone (log 189–193); see pass_public. */

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
