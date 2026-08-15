/*
 * mgl_sync.h
 * MGL
 *
 * Synchronization Layer Subsystem.
 *
 * Bridges the semantic gap between OpenGL framebuffer/sync semantics and
 * Metal render-pass / command-buffer semantics.  Covers several
 * spec-compliance areas:
 *
 *   - Render-pass attachment subresource matching: GL framebuffer
 *     attachments carry (level, layer, textarget) triples that must be
 *     translated to Metal (level, slice, depthPlane) for render-pass
 *     load/store action decisions.
 *   - Metal enum naming (MTLCommandBufferStatus / MTLLoadAction /
 *     MTLStoreAction) for trace logging.
 *   - Render-pass color-texture usage query: determine whether a texture
 *     is bound as a color attachment in a render-pass descriptor.
 *
 * This module is pure specification-compliance machinery: every OpenGL
 * program that uses framebuffer attachments needs these translations when
 * running on Metal, regardless of application.
 */

#ifndef MGL_SYNC_H
#define MGL_SYNC_H

#include "glm_context.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* === Render-pass attachment subresource ===
 *
 * Mirrors the (level, slice, depthPlane) triple that Metal uses to address
 * a subresource of a 2D/array/cube/3D texture in a render-pass attachment. */
typedef struct MGLMetalAttachmentSubresource_t {
    NSUInteger level;
    NSUInteger slice;
    NSUInteger depthPlane;
} MGLMetalAttachmentSubresource;

/* Translate a GL FBOAttachment (level, layer, textarget) to the Metal
 * (level, slice, depthPlane) triple.  Cube-map faces map to slices 0-5;
 * array targets map layer→slice; 3D maps layer→depthPlane. */
MGLMetalAttachmentSubresource mglMetalAttachmentSubresourceForAttachment(const FBOAttachment *attachment);

/* Returns YES if the Metal color attachment descriptor addresses the same
 * subresource as `subresource`. */
BOOL mglMetalRenderPassColorAttachmentMatchesSubresource(MTLRenderPassColorAttachmentDescriptor *descriptor,
                                                         MGLMetalAttachmentSubresource subresource);

/* Returns YES if the Metal depth attachment descriptor addresses the same
 * subresource as `subresource`. */
BOOL mglMetalRenderPassDepthAttachmentMatchesSubresource(MTLRenderPassDepthAttachmentDescriptor *descriptor,
                                                         MGLMetalAttachmentSubresource subresource);

/* Returns YES if the Metal stencil attachment descriptor addresses the same
 * subresource as `subresource`. */
BOOL mglMetalRenderPassStencilAttachmentMatchesSubresource(MTLRenderPassStencilAttachmentDescriptor *descriptor,
                                                           MGLMetalAttachmentSubresource subresource);

/* === Metal enum naming (for trace logging) === */

const char *mglCommandBufferStatusName(MTLCommandBufferStatus status);
const char *mglLoadActionName(MTLLoadAction action);
const char *mglStoreActionName(MTLStoreAction action);

#ifdef __cplusplus
}
#endif

#endif /* MGL_SYNC_H */
