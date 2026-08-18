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
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* === Render-pass attachment subresource ===
 *
 * Mirrors the (level, slice, depthPlane) triple that Metal uses to address
 * a subresource of a 2D/array/cube/3D texture in a render-pass attachment. */
typedef struct MGLMetalAttachmentSubresource_t {
    uint64_t level;
    uint64_t slice;
    uint64_t depthPlane;
} MGLMetalAttachmentSubresource;

/* Translate a GL FBOAttachment (level, layer, textarget) to the Metal
 * (level, slice, depthPlane) triple.  Cube-map faces map to slices 0-5;
 * array targets map layer→slice; 3D maps layer→depthPlane. */
MGLMetalAttachmentSubresource mglMetalAttachmentSubresourceForAttachment(const FBOAttachment *attachment);

/* Returns YES if the Metal color attachment descriptor addresses the same
 * subresource as `subresource`. */
bool mglMetalRenderPassColorAttachmentMatchesSubresource(const void *descriptor,
                                                         MGLMetalAttachmentSubresource subresource);

/* Returns YES if the Metal depth attachment descriptor addresses the same
 * subresource as `subresource`. */
bool mglMetalRenderPassDepthAttachmentMatchesSubresource(const void *descriptor,
                                                         MGLMetalAttachmentSubresource subresource);

/* Returns YES if the Metal stencil attachment descriptor addresses the same
 * subresource as `subresource`. */
bool mglMetalRenderPassStencilAttachmentMatchesSubresource(const void *descriptor,
                                                           MGLMetalAttachmentSubresource subresource);

/* === Metal enum naming (for trace logging) === */

/* Numeric values are part of the opaque command-buffer value-state ABI. */
typedef enum MGLCommandBufferStatus_t {
    MGL_COMMAND_BUFFER_STATUS_NOT_ENQUEUED = 0,
    MGL_COMMAND_BUFFER_STATUS_ENQUEUED = 1,
    MGL_COMMAND_BUFFER_STATUS_COMMITTED = 2,
    MGL_COMMAND_BUFFER_STATUS_SCHEDULED = 3,
    MGL_COMMAND_BUFFER_STATUS_COMPLETED = 4,
    MGL_COMMAND_BUFFER_STATUS_ERROR = 5,
} MGLCommandBufferStatus;

const char *mglCommandBufferStatusName(uint32_t status);
const char *mglLoadActionName(uint32_t action);
const char *mglStoreActionName(uint32_t action);

#ifdef __cplusplus
}
#endif

#endif /* MGL_SYNC_H */
