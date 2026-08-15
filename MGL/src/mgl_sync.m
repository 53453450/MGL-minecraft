/*
 * mgl_sync.m
 * MGL
 *
 * Implementation of the Synchronization Layer Subsystem.
 *
 * See mgl_sync.h for the architectural rationale.  This module owns the
 * pure spec-compliance helpers for translating OpenGL framebuffer / sync
 * semantics to Metal render-pass / command-buffer semantics:
 *   - Render-pass attachment subresource matching.
 *   - Metal enum naming for trace logging.
 *   - Render-pass color-texture usage query.
 *
 * The helpers here are pure: they do not touch the renderer ivar, the
 * command buffer, or the render encoder.  They operate only on the
 * FBOAttachment / MTLRenderPassDescriptor structures passed in as
 * arguments.
 *
 * External dependencies:
 *   - FBOAttachment type (glm_context.h).
 *   - MAX_COLOR_ATTACHMENTS (glm_limits.h).
 *   - Metal framework for MTLRenderPass* / MTLCommandBufferStatus /
 *     MTLLoadAction / MTLStoreAction.
 */

#import "mgl_sync.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

/* === Render-pass attachment subresource === */

MGLMetalAttachmentSubresource mglMetalAttachmentSubresourceForAttachment(const FBOAttachment *attachment)
{
    MGLMetalAttachmentSubresource subresource = {0u, 0u, 0u};
    if (!attachment) {
        return subresource;
    }

    subresource.level = attachment->level;

    switch (attachment->textarget) {
        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
            subresource.slice = 0u;
            break;
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
            subresource.slice = 1u;
            break;
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
            subresource.slice = 2u;
            break;
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
            subresource.slice = 3u;
            break;
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
            subresource.slice = 4u;
            break;
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            subresource.slice = 5u;
            break;

        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            subresource.slice = attachment->layer;
            break;

        case GL_TEXTURE_3D:
            subresource.depthPlane = attachment->layer;
            break;

        default:
            break;
    }

    return subresource;
}

BOOL mglMetalRenderPassColorAttachmentMatchesSubresource(MTLRenderPassColorAttachmentDescriptor *descriptor,
                                                         MGLMetalAttachmentSubresource subresource)
{
    if (!descriptor) {
        return NO;
    }

    return descriptor.level == subresource.level &&
           descriptor.slice == subresource.slice &&
           descriptor.depthPlane == subresource.depthPlane;
}

BOOL mglMetalRenderPassDepthAttachmentMatchesSubresource(MTLRenderPassDepthAttachmentDescriptor *descriptor,
                                                         MGLMetalAttachmentSubresource subresource)
{
    if (!descriptor) {
        return NO;
    }

    return descriptor.level == subresource.level &&
           descriptor.slice == subresource.slice &&
           descriptor.depthPlane == subresource.depthPlane;
}

BOOL mglMetalRenderPassStencilAttachmentMatchesSubresource(MTLRenderPassStencilAttachmentDescriptor *descriptor,
                                                           MGLMetalAttachmentSubresource subresource)
{
    if (!descriptor) {
        return NO;
    }

    return descriptor.level == subresource.level &&
           descriptor.slice == subresource.slice &&
           descriptor.depthPlane == subresource.depthPlane;
}

/* === Metal enum naming (for trace logging) === */

const char *mglCommandBufferStatusName(MTLCommandBufferStatus status)
{
    switch (status) {
        case MTLCommandBufferStatusNotEnqueued: return "NotEnqueued";
        case MTLCommandBufferStatusEnqueued: return "Enqueued";
        case MTLCommandBufferStatusCommitted: return "Committed";
        case MTLCommandBufferStatusScheduled: return "Scheduled";
        case MTLCommandBufferStatusCompleted: return "Completed";
        case MTLCommandBufferStatusError: return "Error";
        default: return "Unknown";
    }
}

const char *mglLoadActionName(MTLLoadAction action)
{
    switch (action) {
        case MTLLoadActionDontCare: return "DontCare";
        case MTLLoadActionLoad: return "Load";
        case MTLLoadActionClear: return "Clear";
        default: return "Unknown";
    }
}

const char *mglStoreActionName(MTLStoreAction action)
{
    switch (action) {
        case MTLStoreActionDontCare: return "DontCare";
        case MTLStoreActionStore: return "Store";
        case MTLStoreActionMultisampleResolve: return "MSResolve";
        case MTLStoreActionStoreAndMultisampleResolve: return "Store+MSResolve";
        case MTLStoreActionUnknown: return "Unknown";
        default: return "Other";
    }
}