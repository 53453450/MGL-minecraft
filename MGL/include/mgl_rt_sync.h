/*
 * mgl_rt_sync.h
 * MGL
 *
 * Render Target Synchronization Subsystem.
 *
 * Bridges the semantic gap between OpenGL's implicit render→sample visibility
 * model and Metal's explicit encoder/barrier model.
 *
 * Responsibilities owned by this module (decision/gate logic only — the
 * actual Y-flipped copy generation and encoder manipulation stay in
 * MGLRenderer.m because they require access to MTLRenderCommandEncoder and
 * the command buffer ivar):
 *
 *   - Gate: which 2D render targets need a GL-sampled copy maintained?
 *           (mglTextureCanUseGLSampledRenderTargetCopy)
 *   - Gate: is a texture currently an attachment of the active render pass?
 *           (mglTextureIsAttachmentOfFramebuffer)
 *   - Pattern: does the active framebuffer look like a "GL sampled copy"
 *             render target (color0 + optional depth, both 2D RT textures)?
 *             (mglFramebufferLooksLikeGLSampledCopyRenderTarget)
 *
 * The Y-Flip Authority decision (USE_ORIGINAL vs USE_SAMPLED_COPY) lives in
 * mgl_coordinate.h — the two subsystems cooperate but the authority model is
 * owned by Coordinate Compatibility.
 */

#ifndef MGL_RT_SYNC_H
#define MGL_RT_SYNC_H

#include "glm_context.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Returns true if `tex` is a 2D render target that needs a Y-flipped
 * sampled copy maintained by RT Sync.
 *
 * Metal tile memory can be stale when a render target is immediately sampled
 * (read-after-write hazard).  Apply the sampled-copy protection to ALL 2D
 * render targets regardless of size or mipmap state.  The previous
 * size-based gating (32-4096, 2048x2048, 16x16) was a Minecraft-specific
 * heuristic that broke when texture dimensions changed between game
 * versions.  This is now a pure spec-compliance gate: any GL_TEXTURE_2D
 * that is also a render target qualifies. */
static inline bool mglTextureCanUseGLSampledRenderTargetCopy(Texture *tex)
{
    return tex &&
           tex->target == GL_TEXTURE_2D &&
           tex->is_render_target;
}

/* Returns true if `tex` is currently a color or depth attachment of the
 * given framebuffer.  Used to guard lazy Y-flip copy refresh: Metal does
 * not allow reading a texture that is simultaneously a render-pass
 * attachment (read-after-write hazard).  When this returns true the lazy
 * refresh must be deferred (the end_render_pass path will refresh it). */
bool mglTextureIsAttachmentOfFramebuffer(Framebuffer *fbo, Texture *tex);

/* Returns true if `fbo` looks like a "GL sampled copy" render target:
 *   - has color attachment 0
 *   - color0 is a 2D render-target texture (passes
 *     mglTextureCanUseGLSampledRenderTargetCopy)
 *   - optional depth attachment (also a texture)
 *
 * On success, optionally writes the color/depth Texture pointers to
 * *outColor / *outDepth.  Returns false otherwise. */
bool mglFramebufferLooksLikeGLSampledCopyRenderTarget(GLMContext glctx,
                                                      Framebuffer *fbo,
                                                      Texture **outColor,
                                                      Texture **outDepth);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RT_SYNC_H */
