/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

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
 * value-state structures passed in as arguments.
 *
 * External dependencies:
 *   - FBOAttachment type (glm_context.h).
 *   - MAX_COLOR_ATTACHMENTS (glm_limits.h).
 *   - Metal framework for MTLRenderPass* / MGLCommandBufferStatus /
 *     MGLLoadAction / MGLStoreAction.
 */

#import "mgl_sync.h"
#import "mgl_render.h"

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

        case GL_TEXTURE_CUBE_MAP:
            /* glFramebufferTextureLayer stores the texture object's cube
             * target plus the selected face in layer.  Whole-level layered
             * cube attachments also use this target, but always carry layer
             * zero and are normalized again when the pass is configured. */
            if (attachment->layer < _CUBE_MAP_MAX_FACE) {
                subresource.slice = attachment->layer;
            }
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

bool mglMetalRenderPassColorAttachmentMatchesSubresource(const void *descriptor,
                                                         MGLMetalAttachmentSubresource subresource)
{
    return mglRenderPassAttachmentMatchesSubresource(descriptor, &subresource);
}

bool mglMetalRenderPassDepthAttachmentMatchesSubresource(const void *descriptor,
                                                         MGLMetalAttachmentSubresource subresource)
{
    return mglRenderPassAttachmentMatchesSubresource(descriptor, &subresource);
}

bool mglMetalRenderPassStencilAttachmentMatchesSubresource(const void *descriptor,
                                                           MGLMetalAttachmentSubresource subresource)
{
    return mglRenderPassAttachmentMatchesSubresource(descriptor, &subresource);
}

/* === Metal enum naming (for trace logging) === */

const char *mglCommandBufferStatusName(uint32_t status)
{
    return mglRenderCommandBufferStatusName(status);
}

const char *mglLoadActionName(uint32_t action)
{
    return mglRenderLoadActionName(action);
}

const char *mglStoreActionName(uint32_t action)
{
    return mglRenderStoreActionName(action);
}
