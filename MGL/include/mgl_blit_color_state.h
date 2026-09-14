/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_blit_color_state.h — the shared state of the mtlBlitFramebuffer color
 * helpers, moved out of MGLRenderer+Blit.m (P0-1, log 136) so the C color paths
 * can take it.  The two `id` fields became opaque handles; the Objective-C
 * side bridges them at its own use sites.
 */

#ifndef MGL_BLIT_COLOR_STATE_H
#define MGL_BLIT_COLOR_STATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <simd/simd.h>

#include "glm_context.h"        /* GLMContext, GLenum */
#include "mgl_sync.h"           /* MGLMetalAttachmentSubresource */
#include "mgl_types_texture.h"  /* Texture */
#include "mgl_types_framebuffer.h" /* Framebuffer, FBOAttachment */
#include "mgl_region_value.h"      /* MGLOriginValue / MGLSizeValue */

#ifdef __cplusplus
extern "C" {
#endif

/* Shared state for mtlBlitFramebuffer color blit helpers.
 * Filled after attachment resolution and clip computation, then passed to the
 * integer / scaled / direct-copy helpers. */
typedef struct MGLBlitColorState {
    GLMContext glm_ctx;
    Framebuffer *readfbo;
    Framebuffer *drawfbo;
    GLenum filter;
    FBOAttachment *readFBOAttachment;
    FBOAttachment *drawFBOAttachment;
    Texture *readTextureObject;
    Texture *drawTextureObject;
    MGLMetalAttachmentSubresource readSubresource;
    MGLMetalAttachmentSubresource drawSubresource;
    void *readtexid;
    void *drawtexid;
    size_t srcTexW, srcTexH, dstTexW, dstTexH;
    int needsFormatConversionBlit;
    int needsRenderTargetSyncBlit;
    int didMsaaResolve;
    int blitNeedsFlip;
    int needsScaledBlit;
    int srcXForward, srcYForward, dstXForward, dstYForward;
    double srcMinX, srcMaxX, srcMinY, srcMaxY;
    double dstMinX, dstMaxX, dstMinY, dstMaxY;
    double srcW, srcH, dstW, dstH;
    int64_t copySrcX, copySrcY, copyDstX, copyDstY, copyW, copyH;
    int64_t srcMetalY, dstMetalY;
    double scaledDstMetalY;
} MGLBlitColorState;

/* -resolveIntegerMultisampleTexture:toTexture:srcOrigin:dstOrigin:size:reason: */
bool mglBlitResolveIntegerMultisampleTexture(void *renderer, void *source_texture,
                                             void *dest_texture,
                                             MGLOriginValue src_origin,
                                             MGLOriginValue dst_origin,
                                             MGLSizeValue size,
                                             const char *reason);

/* -blitFramebufferIntegerColorWithState: */
bool mglBlitIntegerColorWithState(void *renderer,
                                  const MGLBlitColorState *state);

/* -blitFramebufferDirectColorCopyWithState: */
void mglBlitDirectColorWithState(void *renderer,
                                 const MGLBlitColorState *state);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BLIT_COLOR_STATE_H */
