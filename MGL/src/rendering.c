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
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * rendering.c
 * MGL
 *
 */

#include <mach/mach_vm.h>
#include <mach/mach_init.h>
#include <mach/vm_map.h>
#include <limits.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mgl.h"

#include "pixel_utils.h"
#include "glm_context.h"
#include "draw_command.h"
#include "mgl_safety.h"

#include "mgl_trace_log.h"

static Texture *mglStencilAttachmentTexture(FBOAttachment *attachment)
{
    if (!attachment) {
        return NULL;
    }
    return attachment->textarget == GL_RENDERBUFFER
        ? (attachment->buf.rbo ? attachment->buf.rbo->tex : NULL)
        : attachment->buf.tex;
}

static GLboolean mglEnsureStencilShadow(Texture *texture)
{
    if (!texture || texture->width == 0u || texture->height == 0u) {
        return GL_FALSE;
    }
    if (texture->stencil_shadow &&
        texture->stencil_shadow_width == texture->width &&
        texture->stencil_shadow_height == texture->height) {
        return GL_TRUE;
    }
    free(texture->stencil_shadow);
    texture->stencil_shadow = calloc((size_t)texture->width * texture->height, 1u);
    texture->stencil_shadow_width = texture->stencil_shadow ? texture->width : 0u;
    texture->stencil_shadow_height = texture->stencil_shadow ? texture->height : 0u;
    return texture->stencil_shadow ? GL_TRUE : GL_FALSE;
}

static GLboolean mglEnsureDepthShadow(Texture *texture)
{
    if (!texture || texture->width == 0u || texture->height == 0u) return GL_FALSE;
    if (texture->depth_shadow &&
        texture->depth_shadow_width == texture->width &&
        texture->depth_shadow_height == texture->height) return GL_TRUE;
    free(texture->depth_shadow);
    texture->depth_shadow = calloc((size_t)texture->width * texture->height, sizeof(GLfloat));
    texture->depth_shadow_width = texture->depth_shadow ? texture->width : 0u;
    texture->depth_shadow_height = texture->depth_shadow ? texture->height : 0u;
    return texture->depth_shadow ? GL_TRUE : GL_FALSE;
}

static GLboolean mglEnsureRGB10A2Shadow(Texture *texture)
{
    if (!texture || texture->width == 0u || texture->height == 0u) return GL_FALSE;
    /* The only consumer (glReadPixels BGRA repack) reads the shadow solely
     * for RGB10 formats; allocating and filling it for anything else is
     * dead work on every clear.  Keeps the invariant that only RGB10
     * textures ever carry a color shadow. */
    if (texture->internalformat != GL_RGB10 &&
        texture->internalformat != GL_RGB10_A2 &&
        texture->internalformat != GL_RGB10_A2UI) return GL_FALSE;
    if (texture->rgb10a2_shadow &&
        texture->rgb10a2_shadow_width == texture->width &&
        texture->rgb10a2_shadow_height == texture->height) return GL_TRUE;
    free(texture->rgb10a2_shadow);
    texture->rgb10a2_shadow = calloc((size_t)texture->width * texture->height, 4u);
    texture->rgb10a2_shadow_width = texture->rgb10a2_shadow ? texture->width : 0u;
    texture->rgb10a2_shadow_height = texture->rgb10a2_shadow ? texture->height : 0u;
    return texture->rgb10a2_shadow ? GL_TRUE : GL_FALSE;
}

static void mglUpdateDepthShadowForClear(GLMContext ctx)
{
    Framebuffer *fbo = ctx ? ctx->state.framebuffer : NULL;
    Texture *texture = fbo ? mglStencilAttachmentTexture(&fbo->depth) : NULL;
    /* Readers only consult depth_shadow for non-render-target textures
     * (see the glReadPixels depth path); render-target attachments read
     * back via the GPU, so filling their shadow on every clear is dead
     * work. */
    if (!texture || texture->is_render_target || !mglEnsureDepthShadow(texture)) return;
    GLint x0 = 0, y0 = 0, x1 = (GLint)texture->width, y1 = (GLint)texture->height;
    if (ctx->state.caps.scissor_test) {
        if (x0 < ctx->state.var.scissor_box[0]) x0 = ctx->state.var.scissor_box[0];
        if (y0 < ctx->state.var.scissor_box[1]) y0 = ctx->state.var.scissor_box[1];
        if (x1 > ctx->state.var.scissor_box[0] + ctx->state.var.scissor_box[2])
            x1 = ctx->state.var.scissor_box[0] + ctx->state.var.scissor_box[2];
        if (y1 > ctx->state.var.scissor_box[1] + ctx->state.var.scissor_box[3])
            y1 = ctx->state.var.scissor_box[1] + ctx->state.var.scissor_box[3];
    }
    for (GLint y = y0; y < y1; y++)
        for (GLint x = x0; x < x1; x++)
            texture->depth_shadow[(size_t)y * texture->width + x] =
                (GLfloat)ctx->state.var.depth_clear_value;
}

void mglBlitDepthShadow(GLMContext ctx,
                        GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1,
                        GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1)
{
    Framebuffer *readFBO = ctx ? ctx->state.readbuffer : NULL;
    Framebuffer *drawFBO = ctx ? ctx->state.framebuffer : NULL;
    Texture *source = readFBO ? mglStencilAttachmentTexture(&readFBO->depth) : NULL;
    Texture *destination = drawFBO ? mglStencilAttachmentTexture(&drawFBO->depth) : NULL;
    if (!source || !destination ||
        srcX1 <= srcX0 || srcY1 <= srcY0 || dstX1 <= dstX0 || dstY1 <= dstY0) return;
    if (!source->depth_shadow || source->is_render_target) {
        /* No trustworthy CPU source (shadow missing, or stale because draws
         * never update render-target shadows): drop the destination shadow
         * so readers fall back to the GPU path instead of stale data. */
        free(destination->depth_shadow);
        destination->depth_shadow = NULL;
        destination->depth_shadow_width = 0u;
        destination->depth_shadow_height = 0u;
        return;
    }
    if (!mglEnsureDepthShadow(destination)) return;
    GLint srcW = srcX1 - srcX0, srcH = srcY1 - srcY0;
    GLint dstW = dstX1 - dstX0, dstH = dstY1 - dstY0;
    for (GLint y = dstY0; y < dstY1; y++) {
        for (GLint x = dstX0; x < dstX1; x++) {
            if (ctx->state.caps.scissor_test &&
                (x < ctx->state.var.scissor_box[0] ||
                 y < ctx->state.var.scissor_box[1] ||
                 x >= ctx->state.var.scissor_box[0] + ctx->state.var.scissor_box[2] ||
                 y >= ctx->state.var.scissor_box[1] + ctx->state.var.scissor_box[3])) {
                continue;
            }
            GLint sx = srcX0 + ((x - dstX0) * srcW) / dstW;
            GLint sy = srcY0 + ((y - dstY0) * srcH) / dstH;
            if (x >= 0 && y >= 0 && sx >= 0 && sy >= 0 &&
                x < (GLint)destination->width && y < (GLint)destination->height &&
                sx < (GLint)source->width && sy < (GLint)source->height)
                destination->depth_shadow[(size_t)y * destination->width + x] =
                    source->depth_shadow[(size_t)sy * source->width + sx];
        }
    }
}

static void mglUpdateStencilShadowForClear(GLMContext ctx)
{
    Framebuffer *fbo = ctx ? ctx->state.framebuffer : NULL;
    Texture *texture = fbo ? mglStencilAttachmentTexture(&fbo->stencil) : NULL;
    if (!texture || !mglEnsureStencilShadow(texture)) {
        return;
    }

    GLint x0 = 0;
    GLint y0 = 0;
    GLint x1 = (GLint)texture->width;
    GLint y1 = (GLint)texture->height;
    if (ctx->state.caps.scissor_test) {
        if (x0 < ctx->state.var.scissor_box[0]) x0 = ctx->state.var.scissor_box[0];
        if (y0 < ctx->state.var.scissor_box[1]) y0 = ctx->state.var.scissor_box[1];
        if (x1 > ctx->state.var.scissor_box[0] + ctx->state.var.scissor_box[2])
            x1 = ctx->state.var.scissor_box[0] + ctx->state.var.scissor_box[2];
        if (y1 > ctx->state.var.scissor_box[1] + ctx->state.var.scissor_box[3])
            y1 = ctx->state.var.scissor_box[1] + ctx->state.var.scissor_box[3];
    }
    GLubyte value = (GLubyte)ctx->state.var.stencil_clear_value;
    for (GLint row = y0; row < y1; row++) {
        memset(texture->stencil_shadow + (size_t)row * texture->width + x0,
               value,
               (size_t)(x1 - x0));
    }
}

void mglBlitStencilShadow(GLMContext ctx,
                          GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1,
                          GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1)
{
    Framebuffer *readFBO = ctx ? ctx->state.readbuffer : NULL;
    Framebuffer *drawFBO = ctx ? ctx->state.framebuffer : NULL;
    Texture *source = readFBO ? mglStencilAttachmentTexture(&readFBO->stencil) : NULL;
    Texture *destination = drawFBO ? mglStencilAttachmentTexture(&drawFBO->stencil) : NULL;
    if (!source || !destination || !source->stencil_shadow ||
        !mglEnsureStencilShadow(destination) ||
        srcX1 <= srcX0 || srcY1 <= srcY0 || dstX1 <= dstX0 || dstY1 <= dstY0) {
        return;
    }
    GLint srcW = srcX1 - srcX0;
    GLint srcH = srcY1 - srcY0;
    GLint dstW = dstX1 - dstX0;
    GLint dstH = dstY1 - dstY0;
    for (GLint y = dstY0; y < dstY1; y++) {
        for (GLint x = dstX0; x < dstX1; x++) {
            if (ctx->state.caps.scissor_test &&
                (x < ctx->state.var.scissor_box[0] ||
                 y < ctx->state.var.scissor_box[1] ||
                 x >= ctx->state.var.scissor_box[0] + ctx->state.var.scissor_box[2] ||
                 y >= ctx->state.var.scissor_box[1] + ctx->state.var.scissor_box[3])) {
                continue;
            }
            GLint sourceX = srcX0 + ((x - dstX0) * srcW) / dstW;
            GLint sourceY = srcY0 + ((y - dstY0) * srcH) / dstH;
            if (x >= 0 && y >= 0 &&
                sourceX >= 0 && sourceY >= 0 &&
                x < (GLint)destination->width && y < (GLint)destination->height &&
                sourceX < (GLint)source->width && sourceY < (GLint)source->height) {
                destination->stencil_shadow[(size_t)y * destination->width + x] =
                    source->stencil_shadow[(size_t)sourceY * source->width + sourceX];
            }
        }
    }
}

void mglBlitRGB10A2Shadow(GLMContext ctx,
                          GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1,
                          GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1)
{
    Framebuffer *readFBO = ctx ? ctx->state.readbuffer : NULL;
    Framebuffer *drawFBO = ctx ? ctx->state.framebuffer : NULL;
    if (!readFBO || !drawFBO ||
        ctx->state.read_buffer < GL_COLOR_ATTACHMENT0 ||
        ctx->state.read_buffer >= GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS ||
        srcX1 <= srcX0 || srcY1 <= srcY0 || dstX1 <= dstX0 || dstY1 <= dstY0) return;

    GLuint sourceIndex = ctx->state.read_buffer - GL_COLOR_ATTACHMENT0;
    Texture *source = mglStencilAttachmentTexture(&readFBO->color_attachments[sourceIndex]);
    GLboolean sourceUsable = (source && source->rgb10a2_shadow) ? GL_TRUE : GL_FALSE;

    GLsizei drawBufferCount = ctx->state.draw_buffer_count;
    if (drawBufferCount > MAX_COLOR_ATTACHMENTS) drawBufferCount = MAX_COLOR_ATTACHMENTS;
    for (GLsizei slot = 0; slot < drawBufferCount; slot++) {
        GLenum drawBuffer = ctx->state.draw_buffers[slot];
        if (drawBuffer < GL_COLOR_ATTACHMENT0 ||
            drawBuffer >= GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS) continue;
        Texture *destination =
            mglStencilAttachmentTexture(&drawFBO->color_attachments[drawBuffer - GL_COLOR_ATTACHMENT0]);
        if (!sourceUsable) {
            /* Cannot mirror the blit on the CPU: drop the destination shadow
             * so readers fall back to the GPU path instead of stale data. */
            if (destination && destination->rgb10a2_shadow) {
                free(destination->rgb10a2_shadow);
                destination->rgb10a2_shadow = NULL;
                destination->rgb10a2_shadow_width = 0u;
                destination->rgb10a2_shadow_height = 0u;
            }
            continue;
        }
        if (!mglEnsureRGB10A2Shadow(destination)) continue;

        GLint srcW = srcX1 - srcX0, srcH = srcY1 - srcY0;
        GLint dstW = dstX1 - dstX0, dstH = dstY1 - dstY0;
        for (GLint y = dstY0; y < dstY1; y++) {
            for (GLint x = dstX0; x < dstX1; x++) {
                if (ctx->state.caps.scissor_test &&
                    (x < ctx->state.var.scissor_box[0] ||
                     y < ctx->state.var.scissor_box[1] ||
                     x >= ctx->state.var.scissor_box[0] + ctx->state.var.scissor_box[2] ||
                     y >= ctx->state.var.scissor_box[1] + ctx->state.var.scissor_box[3])) continue;
                GLint sx = srcX0 + ((x - dstX0) * srcW) / dstW;
                GLint sy = srcY0 + ((y - dstY0) * srcH) / dstH;
                if (x < 0 || y < 0 || sx < 0 || sy < 0 ||
                    x >= (GLint)destination->width || y >= (GLint)destination->height ||
                    sx >= (GLint)source->width || sy >= (GLint)source->height) continue;
                memcpy(destination->rgb10a2_shadow + ((size_t)y * destination->width + x) * 4u,
                       source->rgb10a2_shadow + ((size_t)sy * source->width + sx) * 4u,
                       4u);
            }
        }
    }
}

void mglInvalidateColorShadowsForDraw(GLMContext ctx)
{
    Framebuffer *fbo = ctx ? ctx->state.framebuffer : NULL;
    if (!fbo) return;
    for (GLuint attachmentIndex = 0; attachmentIndex < MAX_COLOR_ATTACHMENTS; attachmentIndex++) {
        Texture *texture = mglStencilAttachmentTexture(&fbo->color_attachments[attachmentIndex]);
        if (!texture || !texture->rgb10a2_shadow) continue;
        free(texture->rgb10a2_shadow);
        texture->rgb10a2_shadow = NULL;
        texture->rgb10a2_shadow_width = 0u;
        texture->rgb10a2_shadow_height = 0u;
    }
}

static inline bool mglShouldTraceClearCall(uint64_t callCount)
{
    return (callCount <= 60ull) || ((callCount % 200ull) == 0ull);
}

static GLuint mglSafeDrawFramebufferName(GLMContext ctx)
{
    Framebuffer *fbo;

    if (!ctx)
        return 0u;

    fbo = ctx->state.framebuffer;
    if (!fbo)
        return 0u;

    if (!mglObjectPointerLooksPlausible(fbo) ||
        !mglHashTableContainsData(&ctx->state.framebuffer_table, fbo) ||
        !mglPointerRangeIsReadable(fbo, sizeof(*fbo)))
    {
        return 0u;
    }

    return fbo->name;
}

static inline GLdouble mglClampDepthClearValue(GLdouble depth)
{
    if (depth < 0.0)
        return 0.0;
    if (depth > 1.0)
        return 1.0;
    return depth;
}

static bool mglMulSizeT(size_t a, size_t b, size_t *out)
{
    if (!out)
        return false;
    if (a == 0u || b == 0u)
    {
        *out = 0u;
        return true;
    }
    if (a > (SIZE_MAX / b))
        return false;
    *out = a * b;
    return true;
}

static bool mglAddSizeT(size_t a, size_t b, size_t *out)
{
    if (!out)
        return false;
    if (a > (SIZE_MAX - b))
        return false;
    *out = a + b;
    return true;
}

static bool mglAlignSizeT(size_t value, size_t alignment, size_t *out)
{
    if (!out || alignment == 0u)
        return false;
    size_t rem = value % alignment;
    if (rem == 0u)
    {
        *out = value;
        return true;
    }
    return mglAddSizeT(value, alignment - rem, out);
}

static void mglMarkPackBufferReadPixelsWrite(GLMContext ctx,
                                             Buffer *ptr,
                                             GLintptr offset,
                                             GLsizeiptr size,
                                             const void *dst_ptr)
{
    (void)ctx;

    if (!ptr || size <= 0)
        return;

    ptr->ever_written = GL_TRUE;
    if (offset >= 0 &&
        offset <= ptr->size &&
        size <= (ptr->size - offset))
    {
        GLintptr write_end = offset + size;
        if (ptr->written_min < 0 || offset < ptr->written_min)
            ptr->written_min = offset;
        if (ptr->written_max < 0 || write_end > ptr->written_max)
            ptr->written_max = write_end;
        ptr->has_initialized_data = GL_TRUE;
    }

    ptr->last_init_source = kInitReadPixels;
    ptr->last_write_offset = offset;
    ptr->last_write_size = size;
    ptr->last_write_src_ptr = dst_ptr;
    ptr->last_write_src_hash = 0ull;
    ptr->data.dirty_bits |= DIRTY_BUFFER_DATA;
    if (ctx)
        mglMarkStateDirtyBits(&ctx->state, DIRTY_BUFFER);
}

static GLuint mglMaxDrawBuffers(GLMContext ctx)
{
    GLuint maxDrawBuffers = ctx ? ctx->state.var.max_draw_buffers : 0u;
    if (maxDrawBuffers == 0u || maxDrawBuffers > MAX_COLOR_ATTACHMENTS)
        maxDrawBuffers = MAX_COLOR_ATTACHMENTS;
    return maxDrawBuffers;
}

static GLsizei mglDrawBufferCount(GLMContext ctx)
{
    if (!ctx || ctx->state.draw_buffer_count <= 0)
        return 0;
    if (ctx->state.draw_buffer_count > (GLsizei)MAX_COLOR_ATTACHMENTS)
        return MAX_COLOR_ATTACHMENTS;
    return ctx->state.draw_buffer_count;
}

static GLenum mglDrawBufferAt(GLMContext ctx, GLuint slot)
{
    if (!ctx)
        return GL_NONE;

    GLsizei count = mglDrawBufferCount(ctx);
    if (slot < (GLuint)count)
        return ctx->state.draw_buffers[slot];

    return GL_NONE;
}

static GLboolean mglResolveDrawBufferToColorAttachment(GLMContext ctx, GLenum drawBuffer, GLuint *attachmentIndex)
{
    if (!ctx || drawBuffer == GL_NONE)
        return GL_FALSE;

    if (drawBuffer >= GL_COLOR_ATTACHMENT0 &&
        drawBuffer < (GL_COLOR_ATTACHMENT0 + ctx->state.max_color_attachments) &&
        drawBuffer < (GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS))
    {
        if (attachmentIndex)
            *attachmentIndex = (GLuint)(drawBuffer - GL_COLOR_ATTACHMENT0);
        return GL_TRUE;
    }

    switch (drawBuffer)
    {
        case GL_FRONT:
        case GL_BACK:
        case GL_FRONT_LEFT:
        case GL_FRONT_RIGHT:
        case GL_BACK_LEFT:
        case GL_BACK_RIGHT:
        case GL_LEFT:
        case GL_RIGHT:
        case GL_FRONT_AND_BACK:
            if (attachmentIndex)
                *attachmentIndex = 0u;
            return GL_TRUE;

        default:
            return GL_FALSE;
    }
}

static GLboolean mglResolveDrawBufferSlotToColorAttachment(GLMContext ctx, GLint drawbuffer, GLuint *attachmentIndex)
{
    if (!ctx || drawbuffer < 0 || drawbuffer >= (GLint)mglMaxDrawBuffers(ctx))
        return GL_FALSE;

    return mglResolveDrawBufferToColorAttachment(ctx,
                                                 mglDrawBufferAt(ctx, (GLuint)drawbuffer),
                                                 attachmentIndex);
}

static GLboolean mglColorMaskAllowsAnyWrite(GLMContext ctx, GLuint drawBufferIndex)
{
    if (!ctx || drawBufferIndex >= MAX_COLOR_ATTACHMENTS)
        return GL_FALSE;

    return ctx->state.var.color_writemask[drawBufferIndex][0] ||
           ctx->state.var.color_writemask[drawBufferIndex][1] ||
           ctx->state.var.color_writemask[drawBufferIndex][2] ||
           ctx->state.var.color_writemask[drawBufferIndex][3];
}

static GLubyte mglClearComponentToByte(GLfloat value)
{
    if (!(value > 0.0f)) return 0u;
    if (value >= 1.0f) return 255u;
    return (GLubyte)(value * 255.0f + 0.5f);
}

static void mglUpdateRGB10A2ShadowForClear(GLMContext ctx)
{
    Framebuffer *fbo = ctx ? ctx->state.framebuffer : NULL;
    if (!fbo) return;
    GLsizei drawBufferCount = mglDrawBufferCount(ctx);
    for (GLsizei slot = 0; slot < drawBufferCount; slot++) {
        GLuint attachmentIndex = 0u;
        if (!mglResolveDrawBufferToColorAttachment(ctx, mglDrawBufferAt(ctx, (GLuint)slot),
                                                   &attachmentIndex) ||
            attachmentIndex >= MAX_COLOR_ATTACHMENTS) continue;
        Texture *texture = mglStencilAttachmentTexture(&fbo->color_attachments[attachmentIndex]);
        if (!mglEnsureRGB10A2Shadow(texture)) continue;

        GLint x0 = 0, y0 = 0, x1 = (GLint)texture->width, y1 = (GLint)texture->height;
        if (ctx->state.caps.scissor_test) {
            if (x0 < ctx->state.var.scissor_box[0]) x0 = ctx->state.var.scissor_box[0];
            if (y0 < ctx->state.var.scissor_box[1]) y0 = ctx->state.var.scissor_box[1];
            if (x1 > ctx->state.var.scissor_box[0] + ctx->state.var.scissor_box[2])
                x1 = ctx->state.var.scissor_box[0] + ctx->state.var.scissor_box[2];
            if (y1 > ctx->state.var.scissor_box[1] + ctx->state.var.scissor_box[3])
                y1 = ctx->state.var.scissor_box[1] + ctx->state.var.scissor_box[3];
        }
        GLubyte clear[4] = {
            mglClearComponentToByte(ctx->state.color_clear_value[2]),
            mglClearComponentToByte(ctx->state.color_clear_value[1]),
            mglClearComponentToByte(ctx->state.color_clear_value[0]),
            mglClearComponentToByte(ctx->state.color_clear_value[3])
        };
        for (GLint y = y0; y < y1; y++) {
            for (GLint x = x0; x < x1; x++) {
                GLubyte *pixel = texture->rgb10a2_shadow + ((size_t)y * texture->width + x) * 4u;
                if (ctx->state.var.color_writemask[slot][2]) pixel[0] = clear[0];
                if (ctx->state.var.color_writemask[slot][1]) pixel[1] = clear[1];
                if (ctx->state.var.color_writemask[slot][0]) pixel[2] = clear[2];
                if (ctx->state.var.color_writemask[slot][3]) pixel[3] = clear[3];
            }
        }
    }
}

static void mglMaterializeImmediateClear(GLMContext ctx, GLbitfield mask, const char *source, uint64_t callCount)
{
    static uint64_t s_immediateClearCount = 0;

    if (!ctx ||
        ctx->state.caps.scissor_test ||
        (mask & (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) == 0)
    {
        return;
    }

    uint64_t hit = ++s_immediateClearCount;
    if (hit <= 64ull || (hit % 512ull) == 0ull) {
        mglTraceLogExternal("CLEAR_IMMEDIATE source=%s call=%llu mask=0x%x fbo=%u scissor(test=%d box=%d,%d,%d,%d) depth(write=%d clear=%.6f) dirty=0x%x hit=%llu",
                            source ? source : "unknown",
                            (unsigned long long)callCount,
                            (unsigned)mask,
                            (unsigned)mglSafeDrawFramebufferName(ctx),
                            ctx->state.caps.scissor_test ? 1 : 0,
                            (int)ctx->state.var.scissor_box[0],
                            (int)ctx->state.var.scissor_box[1],
                            (int)ctx->state.var.scissor_box[2],
                            (int)ctx->state.var.scissor_box[3],
                            ctx->state.var.depth_writemask ? 1 : 0,
                            (double)ctx->state.var.depth_clear_value,
                            (unsigned)ctx->state.dirty_bits,
                            (unsigned long long)hit);
    }

    mglRendererClearBuffer(ctx, 0, mask);
    ctx->state.clear_bitmask &= ~mask;
}

static void mglStoreCurrentDrawBufferSelection(GLMContext ctx)
{
    if (!ctx) {
        return;
    }

    Framebuffer *fbo = ctx->state.framebuffer;
    GLenum *drawBuffers = fbo ? fbo->draw_buffers : ctx->state.default_draw_buffers;
    GLsizei *drawBufferCount = fbo ? &fbo->draw_buffer_count : &ctx->state.default_draw_buffer_count;
    GLuint *drawBuffer = fbo ? &fbo->draw_buffer : &ctx->state.default_draw_buffer;

    *drawBuffer = ctx->state.draw_buffer;
    *drawBufferCount = ctx->state.draw_buffer_count;
    for (GLuint i = 0; i < MAX_COLOR_ATTACHMENTS; ++i) {
        drawBuffers[i] = ctx->state.draw_buffers[i];
    }
}

static void mglStoreCurrentReadBufferSelection(GLMContext ctx)
{
    if (!ctx) {
        return;
    }

    if (ctx->state.readbuffer) {
        ctx->state.readbuffer->read_buffer = ctx->state.read_buffer;
    } else {
        ctx->state.default_read_buffer = ctx->state.read_buffer;
    }
}

void mglClear(GLMContext ctx, GLbitfield mask)
{
    static uint64_t s_mglClearCallCount = 0;
    static uint64_t s_scissoredClearCount = 0;
    uint64_t callCount = ++s_mglClearCallCount;

    if (mask & ~(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT))
    {
        fprintf(stderr, "MGL Error: mglClear: invalid mask 0x%x\n", mask);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    // glClear mutates framebuffer contents, so deferred draws must land first.
    mglFlushCommandBuffer(ctx);

    if ((mask & GL_STENCIL_BUFFER_BIT) &&
        (ctx->state.var.stencil_writemask != 0u ||
         ctx->state.var.stencil_back_writemask != 0u)) {
        mglUpdateStencilShadowForClear(ctx);
    }
    if ((mask & GL_DEPTH_BUFFER_BIT) && ctx->state.var.depth_writemask) {
        mglUpdateDepthShadowForClear(ctx);
        ctx->state.query_depth_value = (GLfloat)ctx->state.var.depth_clear_value;
        ctx->state.query_depth_known = GL_TRUE;
    }
    if (mask & GL_COLOR_BUFFER_BIT) {
        mglUpdateRGB10A2ShadowForClear(ctx);
    }

    if (ctx->state.caps.scissor_test) {
        uint64_t hit = ++s_scissoredClearCount;
        if (hit <= 32ull || (hit % 512ull) == 0ull) {
            mglTraceLogExternal("CLEAR_SCISSORED_GL call=%llu hit=%llu mask=0x%x fbo=%u drawBuf=0x%x readBuf=0x%x box=%d,%d,%d,%d colorMask=%d%d%d%d depth(write=%d clear=%.6f) stencilWrite(front=0x%x back=0x%x)",
                                (unsigned long long)callCount,
                                (unsigned long long)hit,
                                (unsigned)mask,
                                (unsigned)mglSafeDrawFramebufferName(ctx),
                                (unsigned)ctx->state.draw_buffer,
                                (unsigned)ctx->state.read_buffer,
                                (int)ctx->state.var.scissor_box[0],
                                (int)ctx->state.var.scissor_box[1],
                                (int)ctx->state.var.scissor_box[2],
                                (int)ctx->state.var.scissor_box[3],
                                ctx->state.var.color_writemask[0][0] ? 1 : 0,
                                ctx->state.var.color_writemask[0][1] ? 1 : 0,
                                ctx->state.var.color_writemask[0][2] ? 1 : 0,
                                ctx->state.var.color_writemask[0][3] ? 1 : 0,
                                ctx->state.var.depth_writemask ? 1 : 0,
                                (double)ctx->state.var.depth_clear_value,
                                (unsigned)ctx->state.var.stencil_writemask,
                                (unsigned)ctx->state.var.stencil_back_writemask);
        }
        mglRendererClearBuffer(ctx, 0, mask);
        if ((mask & GL_STENCIL_BUFFER_BIT) &&
            ctx->state.framebuffer &&
            (ctx->state.var.stencil_writemask != 0u ||
             ctx->state.var.stencil_back_writemask != 0u)) {
            ctx->state.framebuffer->stencil.clear_color[0] =
                (GLfloat)ctx->state.var.stencil_clear_value;
        }
        ctx->state.clear_bitmask = 0;
        mglMarkRendererDirtyBits(&ctx->state,
                                 DIRTY_FBO | DIRTY_STATE | DIRTY_RENDER_STATE);
        return;
    }

    Framebuffer *fbo = ctx->state.framebuffer;
    GLbitfield previousMask = ctx->state.clear_bitmask;
    ctx->state.clear_bitmask = mask;

    if (mask & GL_COLOR_BUFFER_BIT)
    {
        if (fbo)
        {
            GLsizei drawBufferCount = mglDrawBufferCount(ctx);
            for (GLsizei slot = 0; slot < drawBufferCount; ++slot)
            {
                GLuint attachmentIndex = 0u;
                if (mglResolveDrawBufferToColorAttachment(ctx,
                                                          mglDrawBufferAt(ctx, (GLuint)slot),
                                                          &attachmentIndex) &&
                    attachmentIndex < ctx->state.max_color_attachments &&
                    (fbo->color_attachment_bitfield & (1u << attachmentIndex)) &&
                    mglColorMaskAllowsAnyWrite(ctx, (GLuint)slot))
                {
                    FBOAttachment *att = &fbo->color_attachments[attachmentIndex];
                    att->clear_bitmask |= GL_COLOR_BUFFER_BIT;
                    att->clear_color[0] = ctx->state.color_clear_value[0];
                    att->clear_color[1] = ctx->state.color_clear_value[1];
                    att->clear_color[2] = ctx->state.color_clear_value[2];
                    att->clear_color[3] = ctx->state.color_clear_value[3];
                    Texture *clearTex = mglStencilAttachmentTexture(att);
                    if (clearTex && clearTex->name == 8u) {
                        mglTraceLogExternal("PENDING_COLOR_CLEAR_SET tex=%u call=%llu fbo=%u attachment=%u slot=%d drawBuf=0x%x readBuf=0x%x mask=0x%x clearMask=0x%x rgba=(%.3f,%.3f,%.3f,%.3f) scissor(test=%d box=%d,%d,%d,%d) colorMask=%d%d%d%d",
                                            (unsigned)clearTex->name,
                                            (unsigned long long)callCount,
                                            (unsigned)mglSafeDrawFramebufferName(ctx),
                                            (unsigned)attachmentIndex,
                                            (int)slot,
                                            (unsigned)mglDrawBufferAt(ctx, (GLuint)slot),
                                            (unsigned)ctx->state.read_buffer,
                                            (unsigned)mask,
                                            (unsigned)att->clear_bitmask,
                                            att->clear_color[0],
                                            att->clear_color[1],
                                            att->clear_color[2],
                                            att->clear_color[3],
                                            ctx->state.caps.scissor_test ? 1 : 0,
                                            (int)ctx->state.var.scissor_box[0],
                                            (int)ctx->state.var.scissor_box[1],
                                            (int)ctx->state.var.scissor_box[2],
                                            (int)ctx->state.var.scissor_box[3],
                                            ctx->state.var.color_writemask[slot][0] ? 1 : 0,
                                            ctx->state.var.color_writemask[slot][1] ? 1 : 0,
                                            ctx->state.var.color_writemask[slot][2] ? 1 : 0,
                                            ctx->state.var.color_writemask[slot][3] ? 1 : 0);
                    }
                }
            }
        }
        else
        {
            GLenum drawBuffer = mglDrawBufferAt(ctx, 0u);
            if (drawBuffer != GL_NONE && mglColorMaskAllowsAnyWrite(ctx, 0u))
            {
                ctx->state.default_fbo_clear_bitmask |= GL_COLOR_BUFFER_BIT;
                ctx->state.default_clear_color[0] = ctx->state.color_clear_value[0];
                ctx->state.default_clear_color[1] = ctx->state.color_clear_value[1];
                ctx->state.default_clear_color[2] = ctx->state.color_clear_value[2];
                ctx->state.default_clear_color[3] = ctx->state.color_clear_value[3];
            }
        }
    }

    if (mask & GL_DEPTH_BUFFER_BIT)
    {
        if (!ctx->state.var.depth_writemask)
            goto clear_stencil;

        if (fbo)
        {
            fbo->depth.clear_bitmask |= GL_DEPTH_BUFFER_BIT;
            fbo->depth.clear_color[0] = (GLfloat)ctx->state.var.depth_clear_value;
        }
        else
        {
            ctx->state.default_fbo_clear_bitmask |= GL_DEPTH_BUFFER_BIT;
        }
    }

clear_stencil:
    if (mask & GL_STENCIL_BUFFER_BIT)
    {
        if (ctx->state.var.stencil_writemask == 0u &&
            ctx->state.var.stencil_back_writemask == 0u)
            goto clear_done;

        if (fbo)
        {
            fbo->stencil.clear_bitmask |= GL_STENCIL_BUFFER_BIT;
            fbo->stencil.clear_color[0] = (GLfloat)ctx->state.var.stencil_clear_value;
        }
        else
        {
            ctx->state.default_fbo_clear_bitmask |= GL_STENCIL_BUFFER_BIT;
        }
    }

clear_done:
    mglMarkRendererDirtyBits(&ctx->state, DIRTY_FBO | DIRTY_STATE);

    if (mglShouldTraceClearCall(callCount)) {
        mglTraceLogExternal("CLEAR_SET call=%llu mask=0x%x prevMask=0x%x fbo=%u drawBuf=0x%x readBuf=0x%x scissor(test=%d box=%d,%d,%d,%d) clearBits(global=0x%x default=0x%x fboDepth=0x%x) depth(write=%d clear=%.6f) dirty=0x%x",
                            (unsigned long long)callCount,
                            (unsigned)mask,
                            (unsigned)previousMask,
                            (unsigned)mglSafeDrawFramebufferName(ctx),
                            (unsigned)ctx->state.draw_buffer,
                            (unsigned)ctx->state.read_buffer,
                            ctx->state.caps.scissor_test ? 1 : 0,
                            (int)ctx->state.var.scissor_box[0],
                            (int)ctx->state.var.scissor_box[1],
                            (int)ctx->state.var.scissor_box[2],
                            (int)ctx->state.var.scissor_box[3],
                            (unsigned)ctx->state.clear_bitmask,
                            (unsigned)ctx->state.default_fbo_clear_bitmask,
                            (unsigned)(ctx->state.framebuffer ? ctx->state.framebuffer->depth.clear_bitmask : 0u),
                            ctx->state.var.depth_writemask ? 1 : 0,
                            (double)ctx->state.var.depth_clear_value,
                            (unsigned)ctx->state.dirty_bits);
    }

    mglMaterializeImmediateClear(ctx, mask, "glClear", callCount);
}

void mglClearColor(GLMContext ctx, GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)
{
    static uint64_t s_mglClearColorCallCount = 0;
    uint64_t callCount = ++s_mglClearColorCallCount;

    ctx->state.color_clear_value[0] = red;
    ctx->state.color_clear_value[1] = green;
    ctx->state.color_clear_value[2] = blue;
    ctx->state.color_clear_value[3] = alpha;

    mglMarkRendererDirtyBits(&ctx->state, DIRTY_STATE);

    if (mglShouldTraceClearCall(callCount)) {
        mglTraceLogExternal("CLEAR_COLOR call=%llu value=(%.3f,%.3f,%.3f,%.3f) fbo=%p(%u) drawBuf=0x%x dirty=0x%x",
                            (unsigned long long)callCount,
                            red,
                            green,
                            blue,
                            alpha,
                            (void *)ctx->state.framebuffer,
                            (unsigned)mglSafeDrawFramebufferName(ctx),
                            (unsigned)ctx->state.draw_buffer,
                            (unsigned)ctx->state.dirty_bits);
    }
}

void mglClearStencil(GLMContext ctx, GLint s)
{
    ctx->state.var.stencil_clear_value = s;

    mglMarkRendererDirtyBits(&ctx->state, DIRTY_STATE);
}

void mglClearDepth(GLMContext ctx, GLdouble depth)
{
    ctx->state.var.depth_clear_value = mglClampDepthClearValue(depth);

    mglMarkRendererDirtyBits(&ctx->state, DIRTY_STATE);

}

void mglClearBufferfv(GLMContext ctx, GLenum buffer, GLint drawbuffer, const GLfloat *value)
{
    static uint64_t s_mglClearBufferfvCallCount = 0;
    uint64_t callCount = ++s_mglClearBufferfvCallCount;
    Framebuffer * fbo = ctx->state.framebuffer;
    FBOAttachment * fboa;

    if (!value)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    switch (buffer) {
        case GL_COLOR:
            if (drawbuffer < 0 || drawbuffer >= (GLint)mglMaxDrawBuffers(ctx))
            {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            mglFlushCommandBuffer(ctx);
            if (fbo) {
                GLuint attachmentIndex = 0u;
                if (mglResolveDrawBufferSlotToColorAttachment(ctx, drawbuffer, &attachmentIndex) &&
                    attachmentIndex < STATE(max_color_attachments) &&
                    (fbo->color_attachment_bitfield & (1u << attachmentIndex)))
                {
                    fboa = &fbo->color_attachments[attachmentIndex];
                    fboa->clear_bitmask |= GL_COLOR_BUFFER_BIT;
                    fboa->clear_color[0] = value[0];
                    fboa->clear_color[1] = value[1];
                    fboa->clear_color[2] = value[2];
                    fboa->clear_color[3] = value[3];
                }
            } else {
                if (drawbuffer != 0)
                    break;
                ctx->state.default_fbo_clear_bitmask |= GL_COLOR_BUFFER_BIT;
                ctx->state.default_clear_color[0] = value[0];
                ctx->state.default_clear_color[1] = value[1];
                ctx->state.default_clear_color[2] = value[2];
                ctx->state.default_clear_color[3] = value[3];
            }
            break;
        case GL_DEPTH:
            if (drawbuffer != 0)
            {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            mglFlushCommandBuffer(ctx);
            if (fbo) {
                fboa = &fbo->depth;
                fboa->clear_bitmask |= GL_DEPTH_BUFFER_BIT;
                fboa->clear_color[0] = (GLfloat)mglClampDepthClearValue(value[0]);
            } else {
                ctx->state.default_fbo_clear_bitmask |= GL_DEPTH_BUFFER_BIT;
                ctx->state.var.depth_clear_value = mglClampDepthClearValue(value[0]);
            }
            /* Keep the depth-shadow CPU readback buffer in sync with this
             * clear, exactly as mglClear() does — glClearBufferfv(GL_DEPTH)
             * must be visible to subsequent glReadPixels(GL_DEPTH_COMPONENT)
             * and glGetTexImage(DEPTH_COMPONENT) reads. */
            if (ctx->state.var.depth_writemask) {
                mglUpdateDepthShadowForClear(ctx);
                ctx->state.query_depth_value = (GLfloat)ctx->state.var.depth_clear_value;
                ctx->state.query_depth_known = GL_TRUE;
            }
            break;
        default:
            fprintf(stderr, "MGL Error: mglClearBufferfv: invalid buffer 0x%x\n", buffer);
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    mglMarkRendererDirtyBits(&ctx->state, DIRTY_FBO | DIRTY_STATE);

    if (mglShouldTraceClearCall(callCount)) {
        mglTraceLogExternal("CLEAR_BUFFERFV call=%llu buffer=0x%x drawbuffer=%d fbo=%u scissor(test=%d box=%d,%d,%d,%d) value=(%.6f,%.6f,%.6f,%.6f) depthWrite=%d dirty=0x%x",
                            (unsigned long long)callCount,
                            (unsigned)buffer,
                            (int)drawbuffer,
                            (unsigned)(fbo ? fbo->name : 0),
                            ctx->state.caps.scissor_test ? 1 : 0,
                            (int)ctx->state.var.scissor_box[0],
                            (int)ctx->state.var.scissor_box[1],
                            (int)ctx->state.var.scissor_box[2],
                            (int)ctx->state.var.scissor_box[3],
                            value ? (double)value[0] : 0.0,
                            value ? (double)value[1] : 0.0,
                            value ? (double)value[2] : 0.0,
                            value ? (double)value[3] : 0.0,
                            ctx->state.var.depth_writemask ? 1 : 0,
                            (unsigned)ctx->state.dirty_bits);
    }

    GLbitfield clearMask = 0;
    if (buffer == GL_COLOR) {
        clearMask = GL_COLOR_BUFFER_BIT;
    } else if (buffer == GL_DEPTH) {
        clearMask = GL_DEPTH_BUFFER_BIT;
    }
    mglMaterializeImmediateClear(ctx, clearMask, "glClearBufferfv", callCount);
}

void mglClearBufferfi(GLMContext ctx, GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil)
{
    static uint64_t s_mglClearBufferfiCallCount = 0;
    uint64_t callCount = ++s_mglClearBufferfiCallCount;
    Framebuffer * fbo = ctx->state.framebuffer;
    FBOAttachment * fboa;

    switch (buffer) {
        case GL_DEPTH_STENCIL:
            if (drawbuffer != 0)
            {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            mglFlushCommandBuffer(ctx);
            if (fbo) {
                fboa = &fbo->depth;
                fboa->clear_bitmask |= GL_DEPTH_BUFFER_BIT;
                fboa->clear_color[0] = (GLfloat)mglClampDepthClearValue(depth);

                fboa = &fbo->stencil;
                fboa->clear_bitmask |= GL_STENCIL_BUFFER_BIT;
                fboa->clear_color[0] = (GLfloat)stencil;
            } else {
                ctx->state.default_fbo_clear_bitmask |= GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT;
                ctx->state.var.depth_clear_value = mglClampDepthClearValue(depth);
                ctx->state.var.stencil_clear_value = (GLuint)stencil;
            }
            /* Keep the depth/stencil shadow CPU readback buffers in sync with
             * this clear, exactly as mglClear() does — glClearBufferfi must
             * be visible to subsequent glReadPixels(GL_DEPTH_STENCIL) and
             * glGetTexImage(DEPTH_STENCIL) reads. */
            if (ctx->state.var.depth_writemask) {
                mglUpdateDepthShadowForClear(ctx);
                ctx->state.query_depth_value = (GLfloat)ctx->state.var.depth_clear_value;
                ctx->state.query_depth_known = GL_TRUE;
            }
            if (ctx->state.var.stencil_writemask != 0u ||
                ctx->state.var.stencil_back_writemask != 0u) {
                mglUpdateStencilShadowForClear(ctx);
            }
            break;
        default:
            fprintf(stderr, "MGL Error: mglClearBufferfi: invalid buffer 0x%x\n", buffer);
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    mglMarkRendererDirtyBits(&ctx->state, DIRTY_FBO | DIRTY_STATE);

    if (mglShouldTraceClearCall(callCount)) {
        mglTraceLogExternal("CLEAR_BUFFERFI call=%llu buffer=0x%x drawbuffer=%d fbo=%u scissor(test=%d box=%d,%d,%d,%d) depth=%.6f stencil=%d depthWrite=%d dirty=0x%x",
                            (unsigned long long)callCount,
                            (unsigned)buffer,
                            (int)drawbuffer,
                            (unsigned)(fbo ? fbo->name : 0),
                            ctx->state.caps.scissor_test ? 1 : 0,
                            (int)ctx->state.var.scissor_box[0],
                            (int)ctx->state.var.scissor_box[1],
                            (int)ctx->state.var.scissor_box[2],
                            (int)ctx->state.var.scissor_box[3],
                            (double)depth,
                            (int)stencil,
                            ctx->state.var.depth_writemask ? 1 : 0,
                            (unsigned)ctx->state.dirty_bits);
    }

    mglMaterializeImmediateClear(ctx,
                                 buffer == GL_DEPTH_STENCIL ? (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) : 0,
                                 "glClearBufferfi",
                                 callCount);
}

void mglFinish(GLMContext ctx)
{
    mglTraceLogExternal("MGL: mglFinish called - flushing and waiting for GPU");
    mglFlushCommandBuffer(ctx);
    mglRendererFlush(ctx, true);
}

void mglFlush(GLMContext ctx)
{
    mglFlushCommandBuffer(ctx);
    mglRendererFlush(ctx, false);
}

void mglDrawBuffers(GLMContext ctx, GLsizei n, const GLenum *bufs)
{
    GLboolean changed = GL_FALSE;

    if (n < 0 || n > (GLsizei)mglMaxDrawBuffers(ctx))
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (n > 0 && !bufs)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    GLbitfield seenColorAttachments = 0u;
    Framebuffer *fbo = ctx->state.framebuffer;
    for (GLsizei i = 0; i < n; ++i)
    {
        GLenum buf = bufs[i];
        if (buf == GL_NONE)
            continue;

        if (fbo)
        {
            if (buf < GL_COLOR_ATTACHMENT0 ||
                buf >= (GL_COLOR_ATTACHMENT0 + STATE(max_color_attachments)) ||
                buf >= (GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS))
            {
                ERROR_RETURN(GL_INVALID_ENUM);
                return;
            }

            GLuint attachmentIndex = (GLuint)(buf - GL_COLOR_ATTACHMENT0);
            GLbitfield attachmentBit = (GLbitfield)(1u << attachmentIndex);
            if (seenColorAttachments & attachmentBit)
            {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            seenColorAttachments |= attachmentBit;
            continue;
        }

        if (buf >= GL_COLOR_ATTACHMENT0 &&
            buf < (GL_COLOR_ATTACHMENT0 + STATE(max_color_attachments)))
        {
            if (buf != GL_COLOR_ATTACHMENT0)
            {
                ERROR_RETURN(GL_INVALID_ENUM);
                return;
            }
            continue;
        }

        switch (buf)
        {
            case GL_FRONT:
            case GL_BACK:
            case GL_FRONT_LEFT:
            case GL_FRONT_RIGHT:
            case GL_BACK_LEFT:
            case GL_BACK_RIGHT:
            case GL_LEFT:
            case GL_RIGHT:
            case GL_FRONT_AND_BACK:
                break;

            default:
                ERROR_RETURN(GL_INVALID_ENUM);
                return;
        }
    }

    if (STATE(draw_buffer_count) != n)
        changed = GL_TRUE;
    for (GLsizei i = 0; !changed && i < n; ++i) {
        if (STATE(draw_buffers[i]) != bufs[i])
            changed = GL_TRUE;
    }
    for (GLsizei i = n; !changed && i < (GLsizei)MAX_COLOR_ATTACHMENTS; ++i) {
        if (STATE(draw_buffers[i]) != GL_NONE)
            changed = GL_TRUE;
    }

    if (changed)
        mglFlushPendingDraws(ctx);

    if (changed)
        mglRendererInvalidateRenderPass(ctx);

    for (GLsizei i = 0; i < n; ++i)
        STATE(draw_buffers[i]) = bufs[i];
    for (GLsizei i = n; i < (GLsizei)MAX_COLOR_ATTACHMENTS; ++i)
        STATE(draw_buffers[i]) = GL_NONE;

    STATE(draw_buffer_count) = n;
    STATE(draw_buffer) = (n > 0) ? bufs[0] : GL_NONE;
    mglStoreCurrentDrawBufferSelection(ctx);
    mglMarkStateDirtyBits(ctx->active_state, DIRTY_FBO | DIRTY_STATE | DIRTY_RENDER_STATE);
}

void mglDrawBuffer(GLMContext ctx, GLenum buf)
{
    if (buf == GL_NONE)
    {
        GLboolean changed =
            STATE(draw_buffer) != GL_NONE ||
            STATE(draw_buffer_count) != 1 ||
            STATE(draw_buffers[0]) != GL_NONE;

        for (GLuint i = 1; !changed && i < MAX_COLOR_ATTACHMENTS; ++i) {
            if (STATE(draw_buffers[i]) != GL_NONE)
                changed = GL_TRUE;
        }

        if (changed)
            mglFlushPendingDraws(ctx);

        if (changed)
            mglRendererInvalidateRenderPass(ctx);

        STATE(draw_buffer) = GL_NONE;
        STATE(draw_buffer_count) = 1;
        STATE(draw_buffers[0]) = GL_NONE;
        for (GLuint i = 1; i < MAX_COLOR_ATTACHMENTS; ++i)
            STATE(draw_buffers[i]) = GL_NONE;
        mglStoreCurrentDrawBufferSelection(ctx);
        mglMarkStateDirtyBits(ctx->active_state,
                              DIRTY_FBO | DIRTY_STATE | DIRTY_RENDER_STATE);
        return;
    }

    if ((buf >= GL_COLOR_ATTACHMENT0) &&
        (buf < (GL_COLOR_ATTACHMENT0 + STATE(max_color_attachments))))
    {
        // ok
    }
    else
    switch(buf)
    {
        case GL_FRONT:
        case GL_BACK:
            break;

        case GL_FRONT_LEFT:
        case GL_FRONT_RIGHT:
        case GL_BACK_LEFT:
        case GL_BACK_RIGHT:
        case GL_LEFT:
        case GL_RIGHT:
        case GL_FRONT_AND_BACK:
            // Accept for compatibility; backend may treat them as front/default draw target.
            break;

        default:
            fprintf(stderr, "MGL Error: mglDrawBuffer: invalid enum 0x%x\n", buf);
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    if (ctx->state.framebuffer &&
        buf != GL_NONE &&
        (buf < GL_COLOR_ATTACHMENT0 ||
         buf >= (GL_COLOR_ATTACHMENT0 + STATE(max_color_attachments)) ||
         buf >= (GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS)))
    {
        fprintf(stderr, "MGL Error: mglDrawBuffer: non-attachment buffer 0x%x is invalid for user FBO\n", buf);
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    GLboolean changed =
        STATE(draw_buffer) != buf ||
        STATE(draw_buffer_count) != 1 ||
        STATE(draw_buffers[0]) != buf;
    for (GLuint i = 1; !changed && i < MAX_COLOR_ATTACHMENTS; ++i) {
        if (STATE(draw_buffers[i]) != GL_NONE)
            changed = GL_TRUE;
    }

    if (changed)
        mglFlushPendingDraws(ctx);

    if (changed)
        mglRendererInvalidateRenderPass(ctx);

    if ((buf >= GL_COLOR_ATTACHMENT0) &&
        (buf < (GL_COLOR_ATTACHMENT0 + STATE(max_color_attachments))))
    {
        // GL_COLOR_ATTACHMENTi selection on user FBO should be sticky even before
        // attachment validation completes. Resolve readiness later during draw/clear/blit.
        Framebuffer *fbo = ctx->state.framebuffer;
        if (fbo)
        {
            GLuint draw_index = (GLuint)(buf - GL_COLOR_ATTACHMENT0);
            FBOAttachment *att = &fbo->color_attachments[draw_index];
            if (att->buf.rbo)
                att->buf.rbo->is_draw_buffer = GL_TRUE;
        }
    }

    STATE(draw_buffer) = buf;
    STATE(draw_buffer_count) = 1;
    STATE(draw_buffers[0]) = buf;
    for (GLuint i = 1; i < MAX_COLOR_ATTACHMENTS; ++i)
        STATE(draw_buffers[i]) = GL_NONE;
    mglStoreCurrentDrawBufferSelection(ctx);
    mglMarkStateDirtyBits(ctx->active_state,
                          DIRTY_FBO | DIRTY_STATE | DIRTY_RENDER_STATE);
}

void mglReadBuffer(GLMContext ctx, GLenum buf)
{
    Framebuffer *readFbo = ctx->state.readbuffer;

    if ((buf >= GL_COLOR_ATTACHMENT0) &&
        (buf < (GL_COLOR_ATTACHMENT0 + STATE(max_color_attachments))))
    {
        // ok
    }
    else
    switch(buf)
    {
        case GL_FRONT:
        case GL_BACK:
        case GL_NONE:
        case GL_FRONT_LEFT:
        case GL_FRONT_RIGHT:
        case GL_BACK_LEFT:
        case GL_BACK_RIGHT:
        case GL_LEFT:
        case GL_RIGHT:
            // These read buffer modes are accepted but may not be fully implemented
            break;

        default:
            fprintf(stderr, "MGL Error: mglReadBuffer: invalid enum 0x%x\n", buf);
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    if (readFbo &&
        buf != GL_NONE &&
        (buf < GL_COLOR_ATTACHMENT0 ||
         buf >= (GL_COLOR_ATTACHMENT0 + STATE(max_color_attachments)) ||
         buf >= (GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS)))
    {
        /* On a user framebuffer the GL spec accepts the non-attachment tokens
         * GL_FRONT/BACK/LEFT/RIGHT/FRONT_LEFT/etc. (they simply select no
         * readable color buffer) — only genuinely unknown values are an
         * error, and those are already rejected above.  Drop the spurious
         * GL_INVALID_OPERATION so restoring a saved GL_READ_BUFFER of GL_BACK
         * onto a depth/stencil-only FBO succeeds. */
        if (buf != GL_FRONT && buf != GL_BACK &&
            buf != GL_FRONT_LEFT && buf != GL_FRONT_RIGHT &&
            buf != GL_BACK_LEFT && buf != GL_BACK_RIGHT &&
            buf != GL_LEFT && buf != GL_RIGHT)
        {
            fprintf(stderr, "MGL Error: mglReadBuffer: non-attachment buffer 0x%x is invalid for user FBO\n", buf);
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
    }

    if ((buf >= GL_COLOR_ATTACHMENT0) &&
        (buf < (GL_COLOR_ATTACHMENT0 + STATE(max_color_attachments))))
    {
        // probably should validate current fbo..
    }

    if (STATE(read_buffer) != buf) {
        /*
         * GL_READ_BUFFER only selects the source for explicit read/blit/copy
         * operations. It is not part of draw framebuffer render-pass identity.
         */
        STATE(read_buffer) = buf;
        mglStoreCurrentReadBufferSelection(ctx);
    }
}

void mglPixelStorei(GLMContext ctx, GLenum pname, GLint param)
{
    switch(pname)
    {
        case GL_PACK_SWAP_BYTES:
            ctx->state.pack.swap_bytes = (param != 0 ? true : false);
            break;

        case GL_PACK_LSB_FIRST:
            ctx->state.pack.lsb_first = (param != 0 ? true : false);
            break;

        case GL_PACK_ROW_LENGTH:
            if (param < 0) {
                fprintf(stderr, "MGL Error: mglPixelStorei: negative PACK_ROW_LENGTH %d\n", param);
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            ctx->state.pack.row_length = param;
            break;

        case GL_PACK_IMAGE_HEIGHT:
            if (param < 0) {
                fprintf(stderr, "MGL Error: mglPixelStorei: negative PACK_IMAGE_HEIGHT %d\n", param);
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            ctx->state.pack.image_height = param;
            break;

        case GL_PACK_SKIP_ROWS:
            if (param < 0) {
                fprintf(stderr, "MGL Error: mglPixelStorei: negative PACK_SKIP_ROWS %d\n", param);
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            ctx->state.pack.skip_rows = param;
            break;

        case GL_PACK_SKIP_PIXELS:
            if (param < 0) {
                fprintf(stderr, "MGL Error: mglPixelStorei: negative PACK_SKIP_PIXELS %d\n", param);
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            ctx->state.pack.skip_pixels = param;
            break;

        case GL_PACK_SKIP_IMAGES:
            if (param < 0) {
                fprintf(stderr, "MGL Error: mglPixelStorei: negative PACK_SKIP_IMAGES %d\n", param);
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            ctx->state.pack.skip_images = param;
            break;

        case GL_PACK_ALIGNMENT:
            switch(param)
            {
                case 1:
                case 2:
                case 4:
                case 8:
                    ctx->state.pack.alignment = param;
                    break;

                default:
                    fprintf(stderr, "MGL Error: mglPixelStorei: invalid PACK_ALIGNMENT %d\n", param);
                    ERROR_RETURN(GL_INVALID_VALUE);
                    break;
            }
            break;
        case GL_PACK_COMPRESSED_BLOCK_WIDTH:
            if (param < 0) { ERROR_RETURN(GL_INVALID_VALUE); return; }
            ctx->state.pack.compressed_block_width = param;
            break;
        case GL_PACK_COMPRESSED_BLOCK_HEIGHT:
            if (param < 0) { ERROR_RETURN(GL_INVALID_VALUE); return; }
            ctx->state.pack.compressed_block_height = param;
            break;
        case GL_PACK_COMPRESSED_BLOCK_DEPTH:
            if (param < 0) { ERROR_RETURN(GL_INVALID_VALUE); return; }
            ctx->state.pack.compressed_block_depth = param;
            break;
        case GL_PACK_COMPRESSED_BLOCK_SIZE:
            if (param < 0) { ERROR_RETURN(GL_INVALID_VALUE); return; }
            ctx->state.pack.compressed_block_size = param;
            break;

        case GL_UNPACK_SWAP_BYTES:
            ctx->state.unpack.swap_bytes = (param != 0 ? true : false);
            break;

        case GL_UNPACK_LSB_FIRST:
            ctx->state.unpack.lsb_first = (param != 0 ? true : false);
            break;

        case GL_UNPACK_ROW_LENGTH:
            if (param < 0) {
                fprintf(stderr, "MGL Error: mglPixelStorei: negative UNPACK_ROW_LENGTH %d\n", param);
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            ctx->state.unpack.row_length = param;
            break;
        case GL_UNPACK_IMAGE_HEIGHT:
            if (param < 0) {
                fprintf(stderr, "MGL Error: mglPixelStorei: negative UNPACK_IMAGE_HEIGHT %d\n", param);
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            ctx->state.unpack.image_height = param;
            break;

        case GL_UNPACK_SKIP_ROWS:
            if (param < 0) {
                fprintf(stderr, "MGL Error: mglPixelStorei: negative UNPACK_SKIP_ROWS %d\n", param);
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            ctx->state.unpack.skip_rows = param;
            break;

        case GL_UNPACK_SKIP_PIXELS:
            if (param < 0) {
                fprintf(stderr, "MGL Error: mglPixelStorei: negative UNPACK_SKIP_PIXELS %d\n", param);
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            ctx->state.unpack.skip_pixels = param;
            break;

        case GL_UNPACK_SKIP_IMAGES:
            if (param < 0) {
                fprintf(stderr, "MGL Error: mglPixelStorei: negative UNPACK_SKIP_IMAGES %d\n", param);
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            ctx->state.unpack.skip_images = param;
            break;

        case GL_UNPACK_ALIGNMENT:
            switch(param)
            {
                case 1:
                case 2:
                case 4:
                case 8:
                    ctx->state.unpack.alignment = param;
                    break;

                default:
                    fprintf(stderr, "MGL Error: mglPixelStorei: invalid UNPACK_ALIGNMENT %d\n", param);
                    ERROR_RETURN(GL_INVALID_VALUE);
                    break;
            }
            break;
        case GL_UNPACK_COMPRESSED_BLOCK_WIDTH:
            if (param < 0) { ERROR_RETURN(GL_INVALID_VALUE); return; }
            ctx->state.unpack.compressed_block_width = param;
            break;
        case GL_UNPACK_COMPRESSED_BLOCK_HEIGHT:
            if (param < 0) { ERROR_RETURN(GL_INVALID_VALUE); return; }
            ctx->state.unpack.compressed_block_height = param;
            break;
        case GL_UNPACK_COMPRESSED_BLOCK_DEPTH:
            if (param < 0) { ERROR_RETURN(GL_INVALID_VALUE); return; }
            ctx->state.unpack.compressed_block_depth = param;
            break;
        case GL_UNPACK_COMPRESSED_BLOCK_SIZE:
            if (param < 0) { ERROR_RETURN(GL_INVALID_VALUE); return; }
            ctx->state.unpack.compressed_block_size = param;
            break;

        default:
            fprintf(stderr, "MGL Error: mglPixelStorei: invalid pname 0x%x\n", pname);
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }
}

void mglPixelStoref(GLMContext ctx, GLenum pname, GLfloat param)
{
    mglPixelStorei(ctx, pname, (GLint)param);
}

typedef struct MGLReadPixelsPackLayout_t {
    size_t pixel_size;
    size_t row_copy_bytes;
    size_t row_length_pixels;
    size_t dst_pitch;
    size_t skip_offset_bytes;
    size_t write_span_bytes;
    size_t required_bytes;
} MGLReadPixelsPackLayout;

/* Swap bytes within each multi-byte element in the readback output buffer.
 * Mirrors GL_PACK_SWAP_BYTES semantics and the CTS swapBytes() helper: each
 * element datum of element_size bytes has its bytes reversed. */
static void mglSwapReadPixelsOutput(uint8_t *data, size_t total_bytes, size_t element_size)
{
    if (!data || element_size <= 1u || total_bytes < element_size) {
        return;
    }
    size_t count = total_bytes / element_size;
    for (size_t i = 0u; i < count; i++) {
        uint8_t *p = data + (i * element_size);
        for (size_t j = 0u; j < element_size / 2u; j++) {
            uint8_t tmp = p[j];
            p[j] = p[element_size - 1u - j];
            p[element_size - 1u - j] = tmp;
        }
    }
}

static bool mglComputeReadPixelsPackLayout(GLMContext ctx,
                                           GLsizei width,
                                           GLsizei height,
                                           size_t pixel_size,
                                           MGLReadPixelsPackLayout *layout)
{
    if (!ctx || !layout || pixel_size == 0u || width <= 0 || height <= 0)
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);

    memset(layout, 0, sizeof(*layout));
    layout->pixel_size = pixel_size;

    if (STATE(pack.row_length) < 0 ||
        STATE(pack.skip_rows) < 0 ||
        STATE(pack.skip_pixels) < 0 ||
        STATE(pack.image_height) < 0 ||
        STATE(pack.skip_images) < 0)
    {
        fprintf(stderr,
                "MGL Error: mglReadPixels: invalid negative pack state rowLength=%d imageHeight=%d skipRows=%d skipPixels=%d skipImages=%d\n",
                STATE(pack.row_length),
                STATE(pack.image_height),
                STATE(pack.skip_rows),
                STATE(pack.skip_pixels),
                STATE(pack.skip_images));
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    layout->row_length_pixels = STATE(pack.row_length) > 0 ?
                                (size_t)STATE(pack.row_length) :
                                (size_t)width;

    size_t unaligned_pitch = 0u;
    if (!mglMulSizeT(layout->row_length_pixels, pixel_size, &unaligned_pitch) ||
        !mglMulSizeT((size_t)width, pixel_size, &layout->row_copy_bytes))
    {
        fprintf(stderr,
                "MGL Error: mglReadPixels: pack row computation overflow rowLength=%zu width=%d pixelSize=%zu\n",
                layout->row_length_pixels,
                width,
                pixel_size);
        ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
    }

    size_t alignment = (size_t)(STATE(pack.alignment) > 0 ? STATE(pack.alignment) : 1);
    if (!mglAlignSizeT(unaligned_pitch, alignment, &layout->dst_pitch))
    {
        fprintf(stderr,
                "MGL Error: mglReadPixels: pack row alignment overflow pitch=%zu alignment=%zu\n",
                unaligned_pitch,
                alignment);
        ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
    }

    size_t skip_rows_bytes = 0u;
    size_t skip_pixels_bytes = 0u;
    if (!mglMulSizeT((size_t)STATE(pack.skip_rows), layout->dst_pitch, &skip_rows_bytes) ||
        !mglMulSizeT((size_t)STATE(pack.skip_pixels), pixel_size, &skip_pixels_bytes) ||
        !mglAddSizeT(skip_rows_bytes, skip_pixels_bytes, &layout->skip_offset_bytes))
    {
        fprintf(stderr,
                "MGL Error: mglReadPixels: pack skip computation overflow skipRows=%d skipPixels=%d dstPitch=%zu pixelSize=%zu\n",
                STATE(pack.skip_rows),
                STATE(pack.skip_pixels),
                layout->dst_pitch,
                pixel_size);
        ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
    }

    layout->write_span_bytes = layout->row_copy_bytes;
    if (height > 1)
    {
        size_t trailing_row_bytes = 0u;
        if (!mglMulSizeT(layout->dst_pitch, (size_t)(height - 1), &trailing_row_bytes) ||
            !mglAddSizeT(layout->write_span_bytes, trailing_row_bytes, &layout->write_span_bytes))
        {
            fprintf(stderr,
                    "MGL Error: mglReadPixels: pack write span overflow dstPitch=%zu height=%d rowBytes=%zu\n",
                    layout->dst_pitch,
                    height,
                    layout->row_copy_bytes);
            ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
        }
    }

    if (!mglAddSizeT(layout->skip_offset_bytes, layout->write_span_bytes, &layout->required_bytes))
    {
        fprintf(stderr,
                "MGL Error: mglReadPixels: pack required byte overflow skip=%zu span=%zu\n",
                layout->skip_offset_bytes,
                layout->write_span_bytes);
        ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
    }

    return true;
}

static bool mglPackRGBA8PixelToFormatType(const uint8_t rgba[4],
                                          uint8_t *dst_pixel,
                                          GLenum format,
                                          GLenum type)
{
    if (!rgba || !dst_pixel) {
        return false;
    }

    int slots = 0;
    int src_idx[4] = {0, 0, 0, 0};
    switch (format) {
        case GL_RGBA: slots = 4; src_idx[0]=0; src_idx[1]=1; src_idx[2]=2; src_idx[3]=3; break;
        case GL_BGRA: slots = 4; src_idx[0]=2; src_idx[1]=1; src_idx[2]=0; src_idx[3]=3; break;
        case GL_RGB:  slots = 3; src_idx[0]=0; src_idx[1]=1; src_idx[2]=2; break;
        case GL_BGR:  slots = 3; src_idx[0]=2; src_idx[1]=1; src_idx[2]=0; break;
        case GL_RG:   slots = 2; src_idx[0]=0; src_idx[1]=1; break;
        case GL_RED:  slots = 1; src_idx[0]=0; break;
        case GL_GREEN: slots = 1; src_idx[0]=1; break;
        case GL_BLUE:  slots = 1; src_idx[0]=2; break;
        case GL_ALPHA: slots = 1; src_idx[0]=3; break;
        /* Integer format aliases - same channel mapping as their non-integer
         * counterparts.  Values are written as raw integers (0-255) rather
         * than normalized, matching CTS expectations for integer readback. */
        case GL_RED_INTEGER:   slots = 1; src_idx[0]=0; break;
        case GL_RG_INTEGER:    slots = 2; src_idx[0]=0; src_idx[1]=1; break;
        case GL_RGB_INTEGER:   slots = 3; src_idx[0]=0; src_idx[1]=1; src_idx[2]=2; break;
        case GL_BGR_INTEGER:   slots = 3; src_idx[0]=2; src_idx[1]=1; src_idx[2]=0; break;
        case GL_RGBA_INTEGER:  slots = 4; src_idx[0]=0; src_idx[1]=1; src_idx[2]=2; src_idx[3]=3; break;
        case GL_BGRA_INTEGER:  slots = 4; src_idx[0]=2; src_idx[1]=1; src_idx[2]=0; src_idx[3]=3; break;
        case 0x8d95 /*GL_GREEN_INTEGER*/: slots = 1; src_idx[0]=1; break;
        case 0x8d96 /*GL_BLUE_INTEGER*/:  slots = 1; src_idx[0]=2; break;
        case 0x8d97 /*GL_ALPHA_INTEGER*/: slots = 1; src_idx[0]=3; break;
        default: return false;
    }

    size_t pixel_size = (size_t)sizeForFormatType(format, type);
    if (pixel_size == 0u) {
        return false;
    }
    memset(dst_pixel, 0, pixel_size);

    if (type == GL_UNSIGNED_BYTE) {
        for (int c = 0; c < slots; c++) {
            dst_pixel[c] = rgba[src_idx[c]];
        }
        return true;
    }

    if (type == GL_FLOAT) {
        float *d = (float *)(void *)dst_pixel;
        for (int c = 0; c < slots; c++) {
            d[c] = (float)rgba[src_idx[c]] / 255.0f;
        }
        return true;
    }

    if (type == GL_HALF_FLOAT) {
        uint16_t *d = (uint16_t *)(void *)dst_pixel;
        for (int c = 0; c < slots; c++) {
            d[c] = mglFloatToHalf((float)rgba[src_idx[c]] / 255.0f);
        }
        return true;
    }

    if (type == GL_BYTE || type == GL_SHORT ||
        type == GL_INT || type == GL_UNSIGNED_INT ||
        type == GL_UNSIGNED_SHORT) {
        size_t comp_size = sizeForType(type);
        for (int c = 0; c < slots; c++) {
            unsigned v = rgba[src_idx[c]];
            uint8_t *out = dst_pixel + (size_t)c * comp_size;
            if (type == GL_BYTE) {
                int8_t iv = (int8_t)((v * 127u + 127u) / 255u);
                memcpy(out, &iv, sizeof(iv));
            } else if (type == GL_SHORT) {
                int16_t iv = (int16_t)(((uint32_t)v * 32767u) / 255u);
                memcpy(out, &iv, sizeof(iv));
            } else if (type == GL_UNSIGNED_SHORT) {
                uint16_t uv = (uint16_t)(v * 257u);
                memcpy(out, &uv, sizeof(uv));
            } else if (type == GL_UNSIGNED_INT) {
                uint32_t uv = (uint32_t)v * 16843009u;
                memcpy(out, &uv, sizeof(uv));
            } else {
                int32_t iv = (int32_t)(((uint64_t)v * 2147483647ull) / 255ull);
                memcpy(out, &iv, sizeof(iv));
            }
        }
        return true;
    }

    uint32_t r = rgba[src_idx[0]];
    uint32_t g = slots > 1 ? rgba[src_idx[1]] : 0u;
    uint32_t b = slots > 2 ? rgba[src_idx[2]] : 0u;
    uint32_t a = slots > 3 ? rgba[src_idx[3]] : 255u;
    if (type == GL_UNSIGNED_BYTE_3_3_2) {
        dst_pixel[0] = (uint8_t)(((r >> 5u) << 5u) | ((g >> 5u) << 2u) | (b >> 6u));
        return true;
    }
    if (type == GL_UNSIGNED_BYTE_2_3_3_REV) {
        dst_pixel[0] = (uint8_t)((r >> 5u) | ((g >> 5u) << 3u) | ((b >> 6u) << 6u));
        return true;
    }
    if (type == GL_UNSIGNED_SHORT_5_6_5 || type == GL_UNSIGNED_SHORT_5_6_5_REV) {
        uint16_t packed = (type == GL_UNSIGNED_SHORT_5_6_5)
            ? (uint16_t)(((r >> 3u) << 11u) | ((g >> 2u) << 5u) | (b >> 3u))
            : (uint16_t)((r >> 3u) | ((g >> 2u) << 5u) | ((b >> 3u) << 11u));
        memcpy(dst_pixel, &packed, sizeof(packed));
        return true;
    }
    if (type == GL_UNSIGNED_SHORT_4_4_4_4 || type == GL_UNSIGNED_SHORT_4_4_4_4_REV) {
        uint16_t packed = (type == GL_UNSIGNED_SHORT_4_4_4_4)
            ? (uint16_t)(((r >> 4u) << 12u) | ((g >> 4u) << 8u) | ((b >> 4u) << 4u) | (a >> 4u))
            : (uint16_t)((r >> 4u) | ((g >> 4u) << 4u) | ((b >> 4u) << 8u) | ((a >> 4u) << 12u));
        memcpy(dst_pixel, &packed, sizeof(packed));
        return true;
    }
    if (type == GL_UNSIGNED_SHORT_5_5_5_1 || type == GL_UNSIGNED_SHORT_1_5_5_5_REV) {
        uint16_t packed = (type == GL_UNSIGNED_SHORT_5_5_5_1)
            ? (uint16_t)(((r >> 3u) << 11u) | ((g >> 3u) << 6u) | ((b >> 3u) << 1u) | (a >= 128u ? 1u : 0u))
            : (uint16_t)((r >> 3u) | ((g >> 3u) << 5u) | ((b >> 3u) << 10u) | ((a >= 128u ? 1u : 0u) << 15u));
        memcpy(dst_pixel, &packed, sizeof(packed));
        return true;
    }
    if (type == GL_UNSIGNED_INT_8_8_8_8 || type == GL_UNSIGNED_INT_8_8_8_8_REV) {
        uint32_t packed = (type == GL_UNSIGNED_INT_8_8_8_8)
            ? ((r << 24u) | (g << 16u) | (b << 8u) | a)
            : (r | (g << 8u) | (b << 16u) | (a << 24u));
        memcpy(dst_pixel, &packed, sizeof(packed));
        return true;
    }
    if (type == GL_UNSIGNED_INT_10_10_10_2 || type == GL_UNSIGNED_INT_2_10_10_10_REV) {
        uint32_t r10 = r * 1023u / 255u;
        uint32_t g10 = g * 1023u / 255u;
        uint32_t b10 = b * 1023u / 255u;
        uint32_t a2 = a * 3u / 255u;
        uint32_t packed = (type == GL_UNSIGNED_INT_10_10_10_2)
            ? ((r10 << 22u) | (g10 << 12u) | (b10 << 2u) | a2)
            : (r10 | (g10 << 10u) | (b10 << 20u) | (a2 << 30u));
        memcpy(dst_pixel, &packed, sizeof(packed));
        return true;
    }
    if (type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
        uint32_t packed = mglPackUnsignedFloatFromUNorm8(r, 6u) |
                          (mglPackUnsignedFloatFromUNorm8(g, 6u) << 11u) |
                          (mglPackUnsignedFloatFromUNorm8(b, 5u) << 22u);
        memcpy(dst_pixel, &packed, sizeof(packed));
        return true;
    }
    if (type == GL_UNSIGNED_INT_5_9_9_9_REV) {
        uint32_t packed = mglPackRGBToSharedExp((double)r / 255.0, (double)g / 255.0, (double)b / 255.0);
        memcpy(dst_pixel, &packed, sizeof(packed));
        return true;
    }

    return false;
}

static bool mglPackRGBA8RowsToFormatType(const uint8_t *src,
                                         size_t src_pitch,
                                         uint8_t *dst,
                                         size_t dst_pitch,
                                         GLsizei width,
                                         GLsizei height,
                                         GLenum format,
                                         GLenum type,
                                         bool source_bgra)
{
    size_t pixel_size = (size_t)sizeForFormatType(format, type);
    if (!src || !dst || width <= 0 || height <= 0 ||
        src_pitch == 0u || dst_pitch == 0u || pixel_size == 0u) {
        return false;
    }

    for (GLsizei y = 0; y < height; y++) {
        const uint8_t *src_row = src + ((size_t)y * src_pitch);
        uint8_t *dst_row = dst + ((size_t)y * dst_pitch);
        for (GLsizei x = 0; x < width; x++) {
            const uint8_t *s = src_row + ((size_t)x * 4u);
            uint8_t rgba[4] = { s[0], s[1], s[2], s[3] };
            if (source_bgra) {
                rgba[0] = s[2];
                rgba[1] = s[1];
                rgba[2] = s[0];
                rgba[3] = s[3];
            }
            if (!mglPackRGBA8PixelToFormatType(rgba,
                                               dst_row + ((size_t)x * pixel_size),
                                               format,
                                               type)) {
                return false;
            }
        }
    }

    return true;
}

static bool mglPackBGRA8ReadPixels(const uint8_t *src,
                                   size_t src_pitch,
                                   uint8_t *dst,
                                   size_t dst_pitch,
                                   GLsizei width,
                                   GLsizei height,
                                   GLenum format,
                                   GLenum type)
{
    if (!src || !dst || width <= 0 || height <= 0 || src_pitch == 0u || dst_pitch == 0u)
        return false;

    if (mglPackRGBA8RowsToFormatType(src,
                                     src_pitch,
                                     dst,
                                     dst_pitch,
                                     width,
                                     height,
                                     format,
                                     type,
                                     true)) {
        return true;
    }

    for (GLsizei y = 0; y < height; y++)
    {
        const uint8_t *src_row = src + ((size_t)y * src_pitch);
        uint8_t *dst_row = dst + ((size_t)y * dst_pitch);

        if (type == GL_FLOAT)
        {
            for (GLsizei x = 0; x < width; x++)
            {
                const uint8_t *s = src_row + ((size_t)x * 4u);
                float *d = (float *)(void *)(dst_row + ((size_t)x * (size_t)sizeForFormatType(format, type)));
                const float r = (float)s[2] / 255.0f;
                const float g = (float)s[1] / 255.0f;
                const float b = (float)s[0] / 255.0f;
                const float a = (float)s[3] / 255.0f;

                switch(format)
                {
                    case GL_BGRA:
                        d[0] = b; d[1] = g; d[2] = r; d[3] = a;
                        break;
                    case GL_RGBA:
                        d[0] = r; d[1] = g; d[2] = b; d[3] = a;
                        break;
                    case GL_BGR:
                        d[0] = b; d[1] = g; d[2] = r;
                        break;
                    case GL_RGB:
                        d[0] = r; d[1] = g; d[2] = b;
                        break;
                    case GL_RG:
                        d[0] = r; d[1] = g;
                        break;
                    case GL_RED:
                        d[0] = r;
                        break;
                    case GL_GREEN:
                        d[0] = g;
                        break;
                    case GL_BLUE:
                        d[0] = b;
                        break;
                    case GL_ALPHA:
                        d[0] = a;
                        break;
                    default:
                        return false;
                }
            }
            continue;
        }

        if (type == GL_HALF_FLOAT)
        {
            for (GLsizei x = 0; x < width; x++)
            {
                const uint8_t *s = src_row + ((size_t)x * 4u);
                uint16_t *d = (uint16_t *)(void *)(dst_row + ((size_t)x * (size_t)sizeForFormatType(format, type)));
                const float r = (float)s[2] / 255.0f;
                const float g = (float)s[1] / 255.0f;
                const float b = (float)s[0] / 255.0f;
                const float a = (float)s[3] / 255.0f;

                switch(format)
                {
                    case GL_BGRA:
                        d[0] = mglFloatToHalf(b); d[1] = mglFloatToHalf(g); d[2] = mglFloatToHalf(r); d[3] = mglFloatToHalf(a);
                        break;
                    case GL_RGBA:
                        d[0] = mglFloatToHalf(r); d[1] = mglFloatToHalf(g); d[2] = mglFloatToHalf(b); d[3] = mglFloatToHalf(a);
                        break;
                    case GL_BGR:
                        d[0] = mglFloatToHalf(b); d[1] = mglFloatToHalf(g); d[2] = mglFloatToHalf(r);
                        break;
                    case GL_RGB:
                        d[0] = mglFloatToHalf(r); d[1] = mglFloatToHalf(g); d[2] = mglFloatToHalf(b);
                        break;
                    case GL_RG:
                        d[0] = mglFloatToHalf(r); d[1] = mglFloatToHalf(g);
                        break;
                    case GL_RED:
                        d[0] = mglFloatToHalf(r);
                        break;
                    case GL_GREEN:
                        d[0] = mglFloatToHalf(g);
                        break;
                    case GL_BLUE:
                        d[0] = mglFloatToHalf(b);
                        break;
                    case GL_ALPHA:
                        d[0] = mglFloatToHalf(a);
                        break;
                    default:
                        return false;
                }
            }
            continue;
        }

        /* Integer scalar types reading from a BGRA8 UNORM framebuffer.
         * GL spec: fixed-point framebuffer data read into an integer type is
         * the fixed-point value scaled to the integer's (unsigned/positive)
         * range and clamped. */
        if (type == GL_BYTE || type == GL_SHORT ||
            type == GL_INT || type == GL_UNSIGNED_INT) {
            size_t comp_size = sizeForType(type);
            size_t dst_comp_bytes = (size_t)sizeForFormatType(format, type);
            for (GLsizei x = 0; x < width; x++) {
                const uint8_t *s = src_row + ((size_t)x * 4u);
                /* BGRA8 source: RGBA = (s[2], s[1], s[0], s[3]) */
                const unsigned cv[4] = { s[2], s[1], s[0], s[3] };
                uint8_t *dst_pixel = dst_row + ((size_t)x * dst_comp_bytes);
                int slots = 0;
                int src_idx[4] = {0, 0, 0, 0};
                switch (format) {
                    case GL_RGBA: slots = 4; src_idx[0]=0; src_idx[1]=1; src_idx[2]=2; src_idx[3]=3; break;
                    case GL_BGRA: slots = 4; src_idx[0]=2; src_idx[1]=1; src_idx[2]=0; src_idx[3]=3; break;
                    case GL_RGB:  slots = 3; src_idx[0]=0; src_idx[1]=1; src_idx[2]=2; break;
                    case GL_BGR:  slots = 3; src_idx[0]=2; src_idx[1]=1; src_idx[2]=0; break;
                    case GL_RG:   slots = 2; src_idx[0]=0; src_idx[1]=1; break;
                    case GL_RED:  slots = 1; src_idx[0]=0; break;
                    case GL_GREEN: slots = 1; src_idx[0]=1; break;
                    case GL_BLUE:  slots = 1; src_idx[0]=2; break;
                    case GL_ALPHA: slots = 1; src_idx[0]=3; break;
                    default: return false;
                }
                for (int c = 0; c < slots; ++c) {
                    unsigned v = cv[src_idx[c]];
                    uint8_t *out = dst_pixel + (size_t)c * comp_size;
                    if (type == GL_BYTE) {
                        int8_t iv = (int8_t)((v * 127u + 127u) / 255u);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_SHORT) {
                        int32_t scaled = (int32_t)((uint32_t)v * 32767u / 255u);
                        if (scaled > 32767) scaled = 32767;
                        int16_t iv = (int16_t)scaled;
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_UNSIGNED_INT) {
                        uint32_t iv = (uint32_t)v * 16843009u; /* 0x01010101 */
                        memcpy(out, &iv, sizeof(iv));
                    } else { /* GL_INT */
                        int32_t scaled = (int32_t)(((uint64_t)v * 2147483647ull) / 255ull);
                        memcpy(out, &scaled, sizeof(scaled));
                    }
                }
            }
            continue;
        }

        switch(format)
        {
            case GL_RGBA:
                if (type == GL_UNSIGNED_SHORT)
                {
                    for (GLsizei x = 0; x < width; x++)
                    {
                        const uint8_t *s = src_row + ((size_t)x * 4u);
                        uint16_t *d = (uint16_t *)(void *)(dst_row + ((size_t)x * 4u * sizeof(uint16_t)));
                        d[0] = (uint16_t)((uint16_t)s[2] * 257u);
                        d[1] = (uint16_t)((uint16_t)s[1] * 257u);
                        d[2] = (uint16_t)((uint16_t)s[0] * 257u);
                        d[3] = (uint16_t)((uint16_t)s[3] * 257u);
                    }
                    break;
                }
                if (type == GL_UNSIGNED_SHORT_4_4_4_4 ||
                    type == GL_UNSIGNED_SHORT_5_5_5_1)
                {
                    for (GLsizei x = 0; x < width; x++)
                    {
                        const uint8_t *s = src_row + ((size_t)x * 4u);
                        uint16_t r = s[2];
                        uint16_t g = s[1];
                        uint16_t b = s[0];
                        uint16_t a = s[3];
                        uint16_t packed = (type == GL_UNSIGNED_SHORT_4_4_4_4)
                            ? (uint16_t)(((r >> 4u) << 12u) |
                                         ((g >> 4u) << 8u) |
                                         ((b >> 4u) << 4u) |
                                         (a >> 4u))
                            : (uint16_t)(((r >> 3u) << 11u) |
                                         ((g >> 3u) << 6u) |
                                         ((b >> 3u) << 1u) |
                                         (a >= 128u ? 1u : 0u));
                        memcpy(dst_row + (size_t)x * sizeof(packed),
                               &packed,
                               sizeof(packed));
                    }
                    break;
                }
                for (GLsizei x = 0; x < width; x++)
                {
                    const uint8_t *s = src_row + ((size_t)x * 4u);
                    uint8_t *d = dst_row + ((size_t)x * 4u);
                    d[0] = s[2];
                    d[1] = s[1];
                    d[2] = s[0];
                    d[3] = s[3];
                }
                break;

            case GL_BGRA:
                memcpy(dst_row, src_row, (size_t)width * 4u);
                break;

            case GL_BGR:
                if (type == GL_UNSIGNED_SHORT)
                {
                    for (GLsizei x = 0; x < width; x++)
                    {
                        const uint8_t *s = src_row + ((size_t)x * 4u);
                        uint16_t *d = (uint16_t *)(void *)(dst_row + ((size_t)x * 3u * sizeof(uint16_t)));
                        d[0] = (uint16_t)((uint16_t)s[0] * 257u);
                        d[1] = (uint16_t)((uint16_t)s[1] * 257u);
                        d[2] = (uint16_t)((uint16_t)s[2] * 257u);
                    }
                    break;
                }
                for (GLsizei x = 0; x < width; x++)
                {
                    const uint8_t *s = src_row + ((size_t)x * 4u);
                    uint8_t *d = dst_row + ((size_t)x * 3u);
                    d[0] = s[0];
                    d[1] = s[1];
                    d[2] = s[2];
                }
                break;

            case GL_RGB:
                if (type == GL_UNSIGNED_SHORT)
                {
                    for (GLsizei x = 0; x < width; x++)
                    {
                        const uint8_t *s = src_row + ((size_t)x * 4u);
                        uint16_t *d = (uint16_t *)(void *)(dst_row + ((size_t)x * 3u * sizeof(uint16_t)));
                        d[0] = (uint16_t)((uint16_t)s[2] * 257u);
                        d[1] = (uint16_t)((uint16_t)s[1] * 257u);
                        d[2] = (uint16_t)((uint16_t)s[0] * 257u);
                    }
                    break;
                }
                for (GLsizei x = 0; x < width; x++)
                {
                    const uint8_t *s = src_row + ((size_t)x * 4u);
                    uint8_t *d = dst_row + ((size_t)x * 3u);
                    d[0] = s[2];
                    d[1] = s[1];
                    d[2] = s[0];
                }
                break;

            case GL_RG:
                if (type == GL_UNSIGNED_SHORT)
                {
                    for (GLsizei x = 0; x < width; x++)
                    {
                        const uint8_t *s = src_row + ((size_t)x * 4u);
                        uint16_t *d = (uint16_t *)(void *)(dst_row + ((size_t)x * 2u * sizeof(uint16_t)));
                        d[0] = (uint16_t)((uint16_t)s[2] * 257u);
                        d[1] = (uint16_t)((uint16_t)s[1] * 257u);
                    }
                    break;
                }
                for (GLsizei x = 0; x < width; x++)
                {
                    const uint8_t *s = src_row + ((size_t)x * 4u);
                    uint8_t *d = dst_row + ((size_t)x * 2u);
                    d[0] = s[2];
                    d[1] = s[1];
                }
                break;

            case GL_RED:
                if (type == GL_UNSIGNED_SHORT)
                {
                    uint16_t *d = (uint16_t *)(void *)dst_row;
                    for (GLsizei x = 0; x < width; x++)
                    {
                        d[x] = (uint16_t)((uint16_t)src_row[((size_t)x * 4u) + 2u] * 257u);
                    }
                    break;
                }
                for (GLsizei x = 0; x < width; x++)
                {
                    dst_row[x] = src_row[((size_t)x * 4u) + 2u];
                }
                break;

            /* Single-channel reads from BGRA8: staging is s[0]=B, s[1]=G, s[2]=R, s[3]=A */
            case GL_GREEN:
                if (type == GL_UNSIGNED_SHORT)
                {
                    uint16_t *d = (uint16_t *)(void *)dst_row;
                    for (GLsizei x = 0; x < width; x++)
                        d[x] = (uint16_t)((uint16_t)src_row[((size_t)x * 4u) + 1u] * 257u);
                    break;
                }
                for (GLsizei x = 0; x < width; x++)
                    dst_row[x] = src_row[((size_t)x * 4u) + 1u];
                break;

            case GL_BLUE:
                if (type == GL_UNSIGNED_SHORT)
                {
                    uint16_t *d = (uint16_t *)(void *)dst_row;
                    for (GLsizei x = 0; x < width; x++)
                        d[x] = (uint16_t)((uint16_t)src_row[((size_t)x * 4u) + 0u] * 257u);
                    break;
                }
                for (GLsizei x = 0; x < width; x++)
                    dst_row[x] = src_row[((size_t)x * 4u) + 0u];
                break;

            case GL_ALPHA:
                if (type == GL_UNSIGNED_SHORT)
                {
                    uint16_t *d = (uint16_t *)(void *)dst_row;
                    for (GLsizei x = 0; x < width; x++)
                        d[x] = (uint16_t)((uint16_t)src_row[((size_t)x * 4u) + 3u] * 257u);
                    break;
                }
                for (GLsizei x = 0; x < width; x++)
                    dst_row[x] = src_row[((size_t)x * 4u) + 3u];
                break;

            default:
                return false;
        }
    }

    return true;
}

static bool mglPackRGBA8ReadPixels(const uint8_t *src,
                                   size_t src_pitch,
                                   uint8_t *dst,
                                   size_t dst_pitch,
                                   GLsizei width,
                                   GLsizei height,
                                   GLenum format,
                                   GLenum type)
{
    if (!src || !dst || width <= 0 || height <= 0 ||
        src_pitch == 0u || dst_pitch == 0u) {
        return false;
    }

    return mglPackRGBA8RowsToFormatType(src,
                                        src_pitch,
                                        dst,
                                        dst_pitch,
                                        width,
                                        height,
                                        format,
                                        type,
                                        false);

    if (type != GL_UNSIGNED_BYTE &&
        type != GL_BYTE &&
        type != GL_UNSIGNED_SHORT &&
        type != GL_SHORT &&
        type != GL_UNSIGNED_INT &&
        type != GL_INT &&
        type != GL_FLOAT &&
        type != GL_HALF_FLOAT &&
        type != GL_UNSIGNED_BYTE_3_3_2 &&
        type != GL_UNSIGNED_BYTE_2_3_3_REV &&
        type != GL_UNSIGNED_SHORT_5_6_5 &&
        type != GL_UNSIGNED_SHORT_5_6_5_REV &&
        type != GL_UNSIGNED_SHORT_4_4_4_4 &&
        type != GL_UNSIGNED_SHORT_4_4_4_4_REV &&
        type != GL_UNSIGNED_SHORT_5_5_5_1 &&
        type != GL_UNSIGNED_SHORT_1_5_5_5_REV &&
        type != GL_UNSIGNED_INT_8_8_8_8 &&
        type != GL_UNSIGNED_INT_8_8_8_8_REV &&
        type != GL_UNSIGNED_INT_10_10_10_2 &&
        type != GL_UNSIGNED_INT_2_10_10_10_REV) {
        return false;
    }

    for (GLsizei y = 0; y < height; y++) {
        const uint8_t *src_row = src + ((size_t)y * src_pitch);
        uint8_t *dst_row = dst + ((size_t)y * dst_pitch);

        if (type != GL_UNSIGNED_BYTE) {
            size_t pixel_size = (size_t)sizeForFormatType(format, type);
            if (pixel_size == 0u) {
                return false;
            }
            for (GLsizei x = 0; x < width; x++) {
                const uint8_t *s = src_row + ((size_t)x * 4u);
                const unsigned cv[4] = { s[0], s[1], s[2], s[3] };
                int slots = 0;
                int src_idx[4] = {0, 0, 0, 0};
                switch (format) {
                    case GL_RGBA: slots = 4; src_idx[0]=0; src_idx[1]=1; src_idx[2]=2; src_idx[3]=3; break;
                    case GL_BGRA: slots = 4; src_idx[0]=2; src_idx[1]=1; src_idx[2]=0; src_idx[3]=3; break;
                    case GL_RGB:  slots = 3; src_idx[0]=0; src_idx[1]=1; src_idx[2]=2; break;
                    case GL_BGR:  slots = 3; src_idx[0]=2; src_idx[1]=1; src_idx[2]=0; break;
                    case GL_RG:   slots = 2; src_idx[0]=0; src_idx[1]=1; break;
                    case GL_RED:  slots = 1; src_idx[0]=0; break;
                    case GL_GREEN: slots = 1; src_idx[0]=1; break;
                    case GL_BLUE:  slots = 1; src_idx[0]=2; break;
                    case GL_ALPHA: slots = 1; src_idx[0]=3; break;
                    default: return false;
                }

                uint8_t *dst_pixel = dst_row + ((size_t)x * pixel_size);
                memset(dst_pixel, 0, pixel_size);
                if (type == GL_FLOAT) {
                    float *d = (float *)(void *)dst_pixel;
                    for (int c = 0; c < slots; c++) {
                        d[c] = (float)cv[src_idx[c]] / 255.0f;
                    }
                } else if (type == GL_HALF_FLOAT) {
                    uint16_t *d = (uint16_t *)(void *)dst_pixel;
                    for (int c = 0; c < slots; c++) {
                        d[c] = mglFloatToHalf((float)cv[src_idx[c]] / 255.0f);
                    }
                } else if (type == GL_BYTE || type == GL_SHORT ||
                           type == GL_INT || type == GL_UNSIGNED_INT ||
                           type == GL_UNSIGNED_SHORT) {
                    size_t comp_size = sizeForType(type);
                    for (int c = 0; c < slots; c++) {
                        unsigned v = cv[src_idx[c]];
                        uint8_t *out = dst_pixel + (size_t)c * comp_size;
                        if (type == GL_BYTE) {
                            int8_t iv = (int8_t)((v * 127u + 127u) / 255u);
                            memcpy(out, &iv, sizeof(iv));
                        } else if (type == GL_SHORT) {
                            int16_t iv = (int16_t)((v * 32767u) / 255u);
                            memcpy(out, &iv, sizeof(iv));
                        } else if (type == GL_UNSIGNED_SHORT) {
                            uint16_t uv = (uint16_t)(v * 257u);
                            memcpy(out, &uv, sizeof(uv));
                        } else if (type == GL_UNSIGNED_INT) {
                            uint32_t uv = (uint32_t)v * 16843009u;
                            memcpy(out, &uv, sizeof(uv));
                        } else {
                            int32_t iv = (int32_t)(((uint64_t)v * 2147483647ull) / 255ull);
                            memcpy(out, &iv, sizeof(iv));
                        }
                    }
                } else {
                    uint32_t r = cv[src_idx[0]];
                    uint32_t g = slots > 1 ? cv[src_idx[1]] : 0u;
                    uint32_t b = slots > 2 ? cv[src_idx[2]] : 0u;
                    uint32_t a = slots > 3 ? cv[src_idx[3]] : 255u;
                    if (type == GL_UNSIGNED_BYTE_3_3_2) {
                        dst_pixel[0] = (uint8_t)(((r >> 5u) << 5u) | ((g >> 5u) << 2u) | (b >> 6u));
                    } else if (type == GL_UNSIGNED_BYTE_2_3_3_REV) {
                        dst_pixel[0] = (uint8_t)((r >> 5u) | ((g >> 5u) << 3u) | ((b >> 6u) << 6u));
                    } else if (type == GL_UNSIGNED_SHORT_5_6_5 || type == GL_UNSIGNED_SHORT_5_6_5_REV) {
                        uint16_t packed = (type == GL_UNSIGNED_SHORT_5_6_5)
                            ? (uint16_t)(((r >> 3u) << 11u) | ((g >> 2u) << 5u) | (b >> 3u))
                            : (uint16_t)((r >> 3u) | ((g >> 2u) << 5u) | ((b >> 3u) << 11u));
                        memcpy(dst_pixel, &packed, sizeof(packed));
                    } else if (type == GL_UNSIGNED_SHORT_4_4_4_4 || type == GL_UNSIGNED_SHORT_4_4_4_4_REV) {
                        uint16_t packed = (type == GL_UNSIGNED_SHORT_4_4_4_4)
                            ? (uint16_t)(((r >> 4u) << 12u) | ((g >> 4u) << 8u) | ((b >> 4u) << 4u) | (a >> 4u))
                            : (uint16_t)((r >> 4u) | ((g >> 4u) << 4u) | ((b >> 4u) << 8u) | ((a >> 4u) << 12u));
                        memcpy(dst_pixel, &packed, sizeof(packed));
                    } else if (type == GL_UNSIGNED_SHORT_5_5_5_1 || type == GL_UNSIGNED_SHORT_1_5_5_5_REV) {
                        uint16_t packed = (type == GL_UNSIGNED_SHORT_5_5_5_1)
                            ? (uint16_t)(((r >> 3u) << 11u) | ((g >> 3u) << 6u) | ((b >> 3u) << 1u) | (a >= 128u ? 1u : 0u))
                            : (uint16_t)((r >> 3u) | ((g >> 3u) << 5u) | ((b >> 3u) << 10u) | ((a >= 128u ? 1u : 0u) << 15u));
                        memcpy(dst_pixel, &packed, sizeof(packed));
                    } else if (type == GL_UNSIGNED_INT_8_8_8_8 || type == GL_UNSIGNED_INT_8_8_8_8_REV) {
                        uint32_t packed = (type == GL_UNSIGNED_INT_8_8_8_8)
                            ? ((r << 24u) | (g << 16u) | (b << 8u) | a)
                            : (r | (g << 8u) | (b << 16u) | (a << 24u));
                        memcpy(dst_pixel, &packed, sizeof(packed));
                    } else if (type == GL_UNSIGNED_INT_10_10_10_2 || type == GL_UNSIGNED_INT_2_10_10_10_REV) {
                        uint32_t r10 = r * 1023u / 255u;
                        uint32_t g10 = g * 1023u / 255u;
                        uint32_t b10 = b * 1023u / 255u;
                        uint32_t a2 = a * 3u / 255u;
                        uint32_t packed = (type == GL_UNSIGNED_INT_10_10_10_2)
                            ? ((r10 << 22u) | (g10 << 12u) | (b10 << 2u) | a2)
                            : (r10 | (g10 << 10u) | (b10 << 20u) | (a2 << 30u));
                        memcpy(dst_pixel, &packed, sizeof(packed));
                    }
                }
            }
            continue;
        }

        switch (format) {
            case GL_RGBA:
                memcpy(dst_row, src_row, (size_t)width * 4u);
                break;
            case GL_BGRA:
                for (GLsizei x = 0; x < width; x++) {
                    const uint8_t *s = src_row + ((size_t)x * 4u);
                    uint8_t *d = dst_row + ((size_t)x * 4u);
                    d[0] = s[2];
                    d[1] = s[1];
                    d[2] = s[0];
                    d[3] = s[3];
                }
                break;
            case GL_RGB:
                for (GLsizei x = 0; x < width; x++) {
                    const uint8_t *s = src_row + ((size_t)x * 4u);
                    uint8_t *d = dst_row + ((size_t)x * 3u);
                    d[0] = s[0];
                    d[1] = s[1];
                    d[2] = s[2];
                }
                break;
            case GL_BGR:
                for (GLsizei x = 0; x < width; x++) {
                    const uint8_t *s = src_row + ((size_t)x * 4u);
                    uint8_t *d = dst_row + ((size_t)x * 3u);
                    d[0] = s[2];
                    d[1] = s[1];
                    d[2] = s[0];
                }
                break;
            case GL_RG:
                for (GLsizei x = 0; x < width; x++) {
                    const uint8_t *s = src_row + ((size_t)x * 4u);
                    uint8_t *d = dst_row + ((size_t)x * 2u);
                    d[0] = s[0];
                    d[1] = s[1];
                }
                break;
            case GL_RED:
                for (GLsizei x = 0; x < width; x++) {
                    dst_row[x] = src_row[(size_t)x * 4u];
                }
                break;
            default:
                return false;
        }
    }

    return true;
}

static GLuint mglFloatReadComponentCount(GLenum format)
{
    switch (format) {
        case GL_RED: return 1u;
        case GL_RG: return 2u;
        case GL_RGB: return 3u;
        case GL_RGBA: return 4u;
        default: return 0u;
    }
}

/* Returns true for non-integer color (non-depth/stencil) read formats. */
static bool mglIsColorReadFormat(GLenum format)
{
    switch (format) {
        case GL_RED:
        case GL_RG:
        case GL_RGB:
        case GL_RGBA:
        case GL_BGR:
        case GL_BGRA:
        case GL_GREEN:
        case GL_BLUE:
        case GL_ALPHA:
            return true;
        default:
            return false;
    }
}

/* Returns true for integer color read formats. */
static bool mglIsIntegerReadFormat(GLenum format)
{
    switch (format) {
        case GL_RED_INTEGER:
        case GL_RG_INTEGER:
        case GL_RGB_INTEGER:
        case GL_RGBA_INTEGER:
        case GL_BGR_INTEGER:
        case GL_BGRA_INTEGER:
        case 0x8d95 /*GL_GREEN_INTEGER*/:
        case 0x8d96 /*GL_BLUE_INTEGER*/:
        case 0x8d97 /*GL_ALPHA_INTEGER*/:
            return true;
        default:
            return false;
    }
}

static bool mglPackR32FFloatReadPixels(const uint8_t *src,
                                       size_t src_pitch,
                                       uint8_t *dst,
                                       size_t dst_pitch,
                                       GLsizei width,
                                       GLsizei height,
                                       GLenum format)
{
    GLuint components = mglFloatReadComponentCount(format);
    if (!src || !dst || width <= 0 || height <= 0 || components == 0u) {
        return false;
    }

    if (format == GL_RED) {
        size_t rowBytes = (size_t)width * sizeof(GLfloat);
        for (GLsizei row = 0; row < height; row++) {
            memcpy(dst + ((size_t)row * dst_pitch),
                   src + ((size_t)row * src_pitch),
                   rowBytes);
        }
        return true;
    }

    for (GLsizei row = 0; row < height; row++) {
        const GLfloat *srcRow = (const GLfloat *)(const void *)(src + ((size_t)row * src_pitch));
        GLfloat *dstRow = (GLfloat *)(void *)(dst + ((size_t)row * dst_pitch));
        for (GLsizei column = 0; column < width; column++) {
            GLfloat *pixel = dstRow + ((size_t)column * components);
            pixel[0] = srcRow[column];
            if (components > 1u) pixel[1] = 0.0f;
            if (components > 2u) pixel[2] = 0.0f;
            if (components > 3u) pixel[3] = 1.0f;
        }
    }

    return true;
}

/* Depth-component readback for mglReadPixels: fetch depth as floats
 * (CPU depth_shadow for non-render-target textures, GPU mtlReadDepthPixels
 * for render targets) then convert to the requested pack type. */
static bool mglReadPixelsDepthComponent(GLMContext ctx,
                                        void *pixels,
                                        const MGLReadPixelsPackLayout *pack_layout,
                                        Buffer *pack_buffer,
                                        GLintptr pack_write_offset,
                                        GLsizeiptr pack_write_size,
                                        GLint x, GLint y,
                                        GLsizei width, GLsizei height,
                                        GLenum format, GLenum type)
{
    Texture *depthTexture = ctx->state.readbuffer
        ? mglStencilAttachmentTexture(&ctx->state.readbuffer->depth)
        : NULL;

    GLfloat *floatDepth = NULL;

    if (depthTexture && depthTexture->depth_shadow &&
        !depthTexture->is_render_target) {
        floatDepth = (GLfloat *)calloc((size_t)width * height, sizeof(GLfloat));
        if (!floatDepth) {
            ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
        }
        for (GLsizei row = 0; row < height; row++) {
            GLint readY = y + row;
            for (GLsizei column = 0; column < width; column++) {
                GLint readX = x + column;
                floatDepth[(size_t)row * width + column] =
                    (readX >= 0 && readY >= 0 &&
                     readX < (GLint)depthTexture->depth_shadow_width &&
                     readY < (GLint)depthTexture->depth_shadow_height)
                    ? depthTexture->depth_shadow[(size_t)readY * depthTexture->depth_shadow_width + readX]
                    : 0.0f;
            }
        }
    } else {
        floatDepth = (GLfloat *)calloc((size_t)width * height, sizeof(GLfloat));
        if (!floatDepth) {
            ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
        }
        mglFlushCommandBuffer(ctx);
        mglRendererReadDepthPixels(ctx,
                                          floatDepth,
                                          (GLuint)(width * sizeof(GLfloat)),
                                          (GLuint)((size_t)width * height * sizeof(GLfloat)),
                                          x, y, width, height);
    }

    if (!floatDepth) {
        static uint64_t s_unsupported_depth_readpixels_count = 0u;
        uint64_t hit = ++s_unsupported_depth_readpixels_count;
        if (hit <= 32u || (hit % 256u) == 0u) {
            fprintf(stderr,
                    "MGL WARNING: mglReadPixels depth readback unavailable format=0x%x type=0x%x hit=%llu\n",
                    format, type, (unsigned long long)hit);
        }
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    for (GLsizei row = 0; row < height; row++) {
        uint8_t *dst = (uint8_t *)pixels + (size_t)row * pack_layout->dst_pitch;
        GLfloat *src = floatDepth + (size_t)row * width;
        for (GLsizei column = 0; column < width; column++) {
            GLfloat d = src[column];
            if (d < 0.0f) d = 0.0f;
            if (d > 1.0f) d = 1.0f;
            switch (type) {
                case GL_FLOAT:
                    ((GLfloat *)dst)[column] = d;
                    break;
                case GL_HALF_FLOAT: {
                    uint16_t h = mglFloatToHalf(d);
                    memcpy(&((uint16_t *)dst)[column], &h, sizeof(uint16_t));
                    break;
                }
                case GL_UNSIGNED_BYTE:
                    ((uint8_t *)dst)[column] = (uint8_t)(d * 255.0f + 0.5f);
                    break;
                case GL_BYTE:
                    ((int8_t *)dst)[column] = (int8_t)(d * 127.0f + 0.5f);
                    break;
                case GL_UNSIGNED_SHORT:
                    ((uint16_t *)dst)[column] = (uint16_t)(d * 65535.0f + 0.5f);
                    break;
                case GL_SHORT:
                    ((int16_t *)dst)[column] = (int16_t)(d * 32767.0f + 0.5f);
                    break;
                case GL_UNSIGNED_INT:
                    ((uint32_t *)dst)[column] = (uint32_t)(d * 4294967295.0f + 0.5f);
                    break;
                case GL_INT:
                    ((int32_t *)dst)[column] = (int32_t)(d * 2147483647.0f + 0.5f);
                    break;
                default:
                    break;
            }
        }
    }

    free(floatDepth);

    if (STATE(pack.swap_bytes) == GL_TRUE) {
        size_t elem_size = mglPixelTypeDatumBytes(type);
        if (elem_size > 1u) {
            mglSwapReadPixelsOutput((uint8_t *)pixels, pack_layout->write_span_bytes, elem_size);
        }
    }

    if (pack_buffer)
        mglMarkPackBufferReadPixelsWrite(ctx, pack_buffer, pack_write_offset, pack_write_size, pixels);
    return true;
}

/* Stencil-index readback for mglReadPixels: per-pixel stencil from the
 * CPU stencil_shadow (kept in sync on clears/blits), converted to the
 * requested pack type. */
static bool mglReadPixelsStencilIndex(GLMContext ctx,
                                      void *pixels,
                                      const MGLReadPixelsPackLayout *pack_layout,
                                      Buffer *pack_buffer,
                                      GLintptr pack_write_offset,
                                      GLsizeiptr pack_write_size,
                                      GLint x, GLint y,
                                      GLsizei width, GLsizei height,
                                      GLenum type)
{
    if (type != GL_UNSIGNED_BYTE && type != GL_BYTE &&
        type != GL_UNSIGNED_SHORT && type != GL_SHORT &&
        type != GL_UNSIGNED_INT && type != GL_INT &&
        type != GL_HALF_FLOAT && type != GL_FLOAT)
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    GLubyte value = (GLubyte)ctx->state.var.stencil_clear_value;
    Texture *stencilTexture = ctx->state.readbuffer
        ? mglStencilAttachmentTexture(&ctx->state.readbuffer->stencil)
        : NULL;
    if (ctx->state.readbuffer &&
        ctx->state.readbuffer->stencil.texture != 0u)
    {
        value = (GLubyte)ctx->state.readbuffer->stencil.clear_color[0];
    }
    for (GLsizei row = 0; row < height; row++)
    {
        uint8_t *dst = (uint8_t *)pixels + (size_t)row * pack_layout->dst_pitch;
        for (GLsizei column = 0; column < width; column++) {
            GLint readX = x + column;
            GLint readY = y + row;
            GLuint stencilValue = (stencilTexture && stencilTexture->stencil_shadow &&
                           readX >= 0 && readY >= 0 &&
                           readX < (GLint)stencilTexture->stencil_shadow_width &&
                           readY < (GLint)stencilTexture->stencil_shadow_height)
                ? stencilTexture->stencil_shadow[(size_t)readY * stencilTexture->stencil_shadow_width + readX]
                : value;
            switch (type) {
                case GL_UNSIGNED_BYTE:
                    dst[column] = (uint8_t)stencilValue;
                    break;
                case GL_BYTE:
                    ((GLbyte *)(void *)dst)[column] = (GLbyte)(GLint)stencilValue;
                    break;
                case GL_UNSIGNED_SHORT:
                    ((GLushort *)(void *)dst)[column] = (GLushort)stencilValue;
                    break;
                case GL_SHORT:
                    ((GLshort *)(void *)dst)[column] = (GLshort)(GLint)stencilValue;
                    break;
                case GL_UNSIGNED_INT:
                    ((GLuint *)(void *)dst)[column] = stencilValue;
                    break;
                case GL_INT:
                    ((GLint *)(void *)dst)[column] = (GLint)stencilValue;
                    break;
                case GL_HALF_FLOAT: {
                    uint16_t h = mglFloatToHalf((float)stencilValue);
                    memcpy(dst + (size_t)column * sizeof(uint16_t), &h, sizeof(uint16_t));
                    break;
                }
                case GL_FLOAT:
                    ((GLfloat *)(void *)dst)[column] = (float)stencilValue;
                    break;
            }
        }
    }
    if (pack_buffer)
        mglMarkPackBufferReadPixelsWrite(ctx, pack_buffer, pack_write_offset, pack_write_size, pixels);
    return true;
}

/* Depth+stencil readback for mglReadPixels: packs interleaved depth/stencil
 * into GL_UNSIGNED_INT_24_8 or GL_FLOAT_32_UNSIGNED_INT_24_8_REV.  Depth uses
 * the CPU depth_shadow (non-render-target) or the GPU mtlReadDepthPixels path
 * (render targets); stencil always reads from stencil_shadow. */
static bool mglReadPixelsDepthStencil(GLMContext ctx,
                                      void *pixels,
                                      const MGLReadPixelsPackLayout *pack_layout,
                                      Buffer *pack_buffer,
                                      GLintptr pack_write_offset,
                                      GLsizeiptr pack_write_size,
                                      GLint x, GLint y,
                                      GLsizei width, GLsizei height,
                                      GLenum type)
{
    Texture *depthTex = ctx->state.readbuffer
        ? mglStencilAttachmentTexture(&ctx->state.readbuffer->depth)
        : NULL;
    Texture *stencilTex = ctx->state.readbuffer
        ? mglStencilAttachmentTexture(&ctx->state.readbuffer->stencil)
        : NULL;

    GLfloat *gpuDepth = NULL;
    GLboolean useGpuDepth = GL_FALSE;

    if (depthTex && depthTex->depth_shadow &&
        !depthTex->is_render_target) {
        /* CPU shadow path — depth_shadow is authoritative for
         * non-render-target depth textures. */
    } else if (depthTex && depthTex->is_render_target) {
        gpuDepth = (GLfloat *)calloc((size_t)width * height, sizeof(GLfloat));
        if (!gpuDepth) {
            ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
        }
        mglFlushCommandBuffer(ctx);
        mglRendererReadDepthPixels(ctx,
                                          gpuDepth,
                                          (GLuint)(width * sizeof(GLfloat)),
                                          (GLuint)((size_t)width * height * sizeof(GLfloat)),
                                          x, y, width, height);
        useGpuDepth = GL_TRUE;
    } else {
        static uint64_t s_unsupported_ds_readpixels_count = 0u;
        uint64_t hit = ++s_unsupported_ds_readpixels_count;
        if (hit <= 32u || (hit % 256u) == 0u) {
            fprintf(stderr,
                    "MGL WARNING: mglReadPixels depth/stencil readback unavailable hit=%llu\n",
                    (unsigned long long)hit);
        }
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    for (GLsizei row = 0; row < height; row++) {
        GLint readY = y + row;
        uint8_t *dst = (uint8_t *)pixels + (size_t)row * pack_layout->dst_pitch;
        for (GLsizei column = 0; column < width; column++) {
            GLint readX = x + column;
            GLfloat depthVal = 0.0f;
            uint8_t stencilVal = 0u;

            if (useGpuDepth) {
                depthVal = gpuDepth[(size_t)row * width + column];
            } else if (depthTex->depth_shadow &&
                       readX >= 0 && readY >= 0 &&
                       readX < (GLint)depthTex->depth_shadow_width &&
                       readY < (GLint)depthTex->depth_shadow_height) {
                depthVal = depthTex->depth_shadow[
                    (size_t)readY * depthTex->depth_shadow_width + readX];
            }
            if (stencilTex && stencilTex->stencil_shadow &&
                readX >= 0 && readY >= 0 &&
                readX < (GLint)stencilTex->stencil_shadow_width &&
                readY < (GLint)stencilTex->stencil_shadow_height) {
                stencilVal = stencilTex->stencil_shadow[
                    (size_t)readY * stencilTex->stencil_shadow_width + readX];
            }

            if (depthVal < 0.0f) depthVal = 0.0f;
            if (depthVal > 1.0f) depthVal = 1.0f;

            if (type == GL_UNSIGNED_INT_24_8) {
                uint32_t packed = ((uint32_t)(depthVal * 16777215.0f + 0.5f) << 8) |
                                  (uint32_t)stencilVal;
                memcpy(dst + (size_t)column * sizeof(uint32_t), &packed, sizeof(uint32_t));
            } else if (type == GL_FLOAT_32_UNSIGNED_INT_24_8_REV) {
                /* 64-bit: float depth, then 32 bits with stencil in low 8.
                 * Use memcpy to avoid unaligned access on arm64 when the
                 * destination buffer is not 8-byte aligned. */
                GLfloat depthPart = depthVal;
                uint32_t stencilPart = (uint32_t)stencilVal;
                uint8_t *dst_pixel = dst + (size_t)column * 8u;
                memcpy(dst_pixel, &depthPart, sizeof(GLfloat));
                memcpy(dst_pixel + 4u, &stencilPart, sizeof(uint32_t));
            }
        }
    }

    if (gpuDepth)
        free(gpuDepth);

    if (STATE(pack.swap_bytes) == GL_TRUE) {
        size_t elem_size = mglPixelTypeDatumBytes(type);
        if (elem_size > 1u) {
            mglSwapReadPixelsOutput((uint8_t *)pixels, pack_layout->write_span_bytes, elem_size);
        }
    }

    if (pack_buffer)
        mglMarkPackBufferReadPixelsWrite(ctx, pack_buffer, pack_write_offset, pack_write_size, pixels);
    return true;
}

void mglReadPixels(GLMContext ctx, GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, void *pixels)
{
    GLuint pixel_size;
    Buffer *pack_buffer = NULL;
    GLintptr pack_write_offset = 0;
    GLsizeiptr pack_write_size = 0;

    pixel_size = sizeForFormatType(format, type);
    // ERROR_CHECK_RETURN(pixel_size != 0, GL_INVALID_ENUM);
    if (pixel_size == 0) {
        fprintf(stderr, "MGL Error: mglReadPixels: invalid format/type combination (format=0x%x type=0x%x)\n", format, type);
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    // ERROR_CHECK_RETURN(width > 0, GL_INVALID_ENUM);
    if (width < 0) {
        fprintf(stderr, "MGL Error: mglReadPixels: width < 0 (%d)\n", width);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    // ERROR_CHECK_RETURN(height > 0, GL_INVALID_ENUM);
    if (height < 0) {
        fprintf(stderr, "MGL Error: mglReadPixels: height < 0 (%d)\n", height);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (width == 0 || height == 0)
    {
        return;
    }

    MGLReadPixelsPackLayout pack_layout;

    switch(format)
    {
        case GL_STENCIL_INDEX:
            if (!ctx->state.readbuffer) {
                ERROR_CHECK_RETURN(ctx->stencil_format.mtl_pixel_format > 0, GL_INVALID_OPERATION);
            } else {
                ERROR_CHECK_RETURN(ctx->state.readbuffer->stencil.texture != 0, GL_INVALID_OPERATION);
            }
            break;

        case GL_DEPTH_COMPONENT:
            if (!ctx->state.readbuffer) {
                ERROR_CHECK_RETURN(ctx->depth_format.mtl_pixel_format > 0, GL_INVALID_OPERATION);
            } else {
                ERROR_CHECK_RETURN(ctx->state.readbuffer->depth.texture != 0, GL_INVALID_OPERATION);
            }
            break;

        case GL_DEPTH_STENCIL:
            if (!ctx->state.readbuffer) {
                ERROR_CHECK_RETURN((ctx->depth_format.mtl_pixel_format > 0) &&
                                   (ctx->stencil_format.mtl_pixel_format > 0), GL_INVALID_OPERATION);
            } else {
                ERROR_CHECK_RETURN(ctx->state.readbuffer->depth.texture != 0 &&
                                   ctx->state.readbuffer->stencil.texture != 0, GL_INVALID_OPERATION);
            }
            switch(type)
            {
                case GL_UNSIGNED_INT_24_8:
                case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
                    break;

                default:
                    /* GL_DEPTH_STENCIL is only packable into UNSIGNED_INT_24_8
                     * or FLOAT_32_UNSIGNED_INT_24_8_REV; any other type is an
                     * enum error. Return immediately so the generic type/format
                     * switch below cannot overwrite this with a spurious
                     * GL_INVALID_OPERATION. (CTS packed_depth_stencil
                     * validate_errors expects GL_INVALID_ENUM here.) */
                    ERROR_RETURN(GL_INVALID_ENUM);
                    return;
            }
            break;

        default:
            break;
    }

    switch(type)
    {
        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
            if (format != GL_RGB && format != GL_RGB_INTEGER) {
                fprintf(stderr, "MGL Error: mglReadPixels: invalid format for type (format=0x%x type=0x%x)\n", format, type);
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            break;

        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
            if (!(format == GL_RGBA || format == GL_BGRA ||
                  format == GL_RGBA_INTEGER || format == GL_BGRA_INTEGER)) {
                fprintf(stderr, "MGL Error: mglReadPixels: invalid format for type (format=0x%x type=0x%x)\n", format, type);
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            break;

        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
            if (format != GL_RGB) {
                fprintf(stderr, "MGL Error: mglReadPixels: invalid format for type (format=0x%x type=0x%x)\n", format, type);
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            break;

        case GL_UNSIGNED_INT_24_8:
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
            if (format != GL_DEPTH_STENCIL) {
                fprintf(stderr, "MGL Error: mglReadPixels: invalid format for type (format=0x%x type=0x%x)\n", format, type);
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            break;
    }

    /* Rule: integer transfer format cannot be paired with float/half-float type.
     * (CTS glcPackedPixelsTests isFormatValid: GL core path.) */
    {
        bool fmt_is_integer = (format == GL_RED_INTEGER   ||
                               format == GL_RG_INTEGER    ||
                               format == GL_RGB_INTEGER   ||
                               format == GL_BGR_INTEGER   ||
                               format == GL_RGBA_INTEGER  ||
                               format == GL_BGRA_INTEGER  ||
                               format == 0x8d95 /*GL_GREEN_INTEGER*/ ||
                               format == 0x8d96 /*GL_BLUE_INTEGER*/  ||
                               format == 0x8d97 /*GL_ALPHA_INTEGER*/);
        if (fmt_is_integer && (type == GL_FLOAT || type == GL_HALF_FLOAT)) {
            fprintf(stderr, "MGL Error: mglReadPixels: integer format with float type (format=0x%x type=0x%x)\n", format, type);
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
    }

    /* Rule: integer transfer format requires an integer framebuffer, and
     * a non-integer transfer format requires a non-integer framebuffer.
     * (CTS glcPackedPixelsTests isFormatValid: GL core path.) */
    {
        bool fmt_is_int = (format == GL_RED_INTEGER   ||
                           format == GL_RG_INTEGER    ||
                           format == GL_RGB_INTEGER   ||
                           format == GL_BGR_INTEGER   ||
                           format == GL_RGBA_INTEGER  ||
                           format == GL_BGRA_INTEGER  ||
                           format == 0x8d95 /*GL_GREEN_INTEGER*/ ||
                           format == 0x8d96 /*GL_BLUE_INTEGER*/  ||
                           format == 0x8d97 /*GL_ALPHA_INTEGER*/);
        /* Determine if the current read framebuffer color attachment is integer */
        bool fb_is_int = false;
        Framebuffer *readFb = ctx->state.readbuffer;
        if (readFb &&
            ctx->state.read_buffer >= GL_COLOR_ATTACHMENT0 &&
            ctx->state.read_buffer < GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS) {
            GLuint idx = ctx->state.read_buffer - GL_COLOR_ATTACHMENT0;
            if (readFb->color_attachments[idx].texture) {
                    FBOAttachment *att = &readFb->color_attachments[idx];
                    GLint ifmt = 0;
                    if (att->textarget == GL_RENDERBUFFER && att->buf.rbo && att->buf.rbo->tex) {
                        ifmt = att->buf.rbo->tex->internalformat;
                    } else if (att->buf.tex) {
                        ifmt = att->buf.tex->internalformat;
                    }
                    fb_is_int = (ifmt == GL_R8I       || ifmt == GL_R8UI      ||
                                 ifmt == GL_R16I      || ifmt == GL_R16UI     ||
                                 ifmt == GL_R32I      || ifmt == GL_R32UI     ||
                                 ifmt == GL_RG8I      || ifmt == GL_RG8UI     ||
                                 ifmt == GL_RG16I     || ifmt == GL_RG16UI    ||
                                 ifmt == GL_RG32I     || ifmt == GL_RG32UI    ||
                                 ifmt == GL_RGB8I     || ifmt == GL_RGB8UI    ||
                                 ifmt == GL_RGB16I    || ifmt == GL_RGB16UI   ||
                                 ifmt == GL_RGB32I    || ifmt == GL_RGB32UI   ||
                                 ifmt == GL_RGBA8I    || ifmt == GL_RGBA8UI   ||
                                 ifmt == GL_RGBA16I   || ifmt == GL_RGBA16UI  ||
                                 ifmt == GL_RGBA32I   || ifmt == GL_RGBA32UI  ||
                                 ifmt == GL_RGB10_A2UI);
            }
        }
        if (fmt_is_int != fb_is_int) {
            fprintf(stderr, "MGL Error: mglReadPixels: integer/non-integer mismatch (format=0x%x fb_int=%d)\n", format, (int)fb_is_int);
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
    }

    if (!mglComputeReadPixelsPackLayout(ctx,
                                        width,
                                        height,
                                        (size_t)pixel_size,
                                        &pack_layout))
    {
        return;
    }

    if (STATE(buffers[_PIXEL_PACK_BUFFER]))
    {
        Buffer *ptr;
        uintptr_t offset;
        uint8_t *base;

        ptr = STATE(buffers[_PIXEL_PACK_BUFFER]);

        if (ptr->mapped) {
            GLboolean persistent_map =
                ((ptr->storage_flags & GL_MAP_PERSISTENT_BIT) != 0u) &&
                ((ptr->access_flags & GL_MAP_PERSISTENT_BIT) != 0u);
            static uint64_t s_mapped_pbo_readpixels_count = 0u;
            uint64_t hit = ++s_mapped_pbo_readpixels_count;

            if (!persistent_map) {
                fprintf(stderr,
                        "MGL Error: mglReadPixels: pixel pack buffer is mapped "
                        "(buffer=%u storageFlags=0x%x access=0x%x accessFlags=0x%x mappedRange=%lld,%lld)\n",
                        ptr->name,
                        ptr->storage_flags,
                        ptr->access,
                        ptr->access_flags,
                        (long long)ptr->mapped_offset,
                        (long long)ptr->mapped_length);
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }

            if (hit <= 32u || (hit % 256u) == 0u) {
                fprintf(stderr,
                        "MGL TRACE ReadPixels.PBO.persistentMapped hit=%llu buffer=%u storageFlags=0x%x accessFlags=0x%x "
                        "mappedRange=%lld,%lld size=%lld offset=%p required=%zu backing=%p\n",
                        (unsigned long long)hit,
                        ptr->name,
                        ptr->storage_flags,
                        ptr->access_flags,
                        (long long)ptr->mapped_offset,
                        (long long)ptr->mapped_length,
                        (long long)ptr->size,
                        pixels,
                        pack_layout.required_bytes,
                        (void *)(uintptr_t)ptr->data.buffer_data);
            }
        }

        if (ptr->size < 0)
        {
            fprintf(stderr, "MGL Error: mglReadPixels: pixel pack buffer has negative size (%ld)\n", (long)ptr->size);
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }

        offset = (uintptr_t)pixels;
        if (pixel_size && (offset % (uintptr_t)pixel_size) != 0)
        {
            fprintf(stderr, "MGL Error: mglReadPixels: pixel pack buffer offset not aligned (offset=%lu pixel_size=%u)\n", (unsigned long)offset, pixel_size);
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }

        if ((size_t)ptr->size < offset || (size_t)ptr->size - offset < pack_layout.required_bytes)
        {
            fprintf(stderr, "MGL Error: mglReadPixels: pixel pack buffer too small (size=%ld offset=%lu req=%zu)\n",
                    (long)ptr->size, (unsigned long)offset, pack_layout.required_bytes);
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }

        base = (uint8_t *)(uintptr_t)ptr->data.buffer_data;
        if (!base)
        {
            fprintf(stderr, "MGL Error: mglReadPixels: pixel pack buffer has no CPU storage\n");
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }

        if (offset > (uintptr_t)INTPTR_MAX ||
            pack_layout.skip_offset_bytes > (size_t)INTPTR_MAX ||
            pack_layout.write_span_bytes > (size_t)INTPTR_MAX ||
            (size_t)offset + pack_layout.skip_offset_bytes > (size_t)INTPTR_MAX)
        {
            fprintf(stderr,
                    "MGL Error: mglReadPixels: pixel pack write bookkeeping overflow "
                    "(offset=%lu skip=%zu span=%zu)\n",
                    (unsigned long)offset,
                    pack_layout.skip_offset_bytes,
                    pack_layout.write_span_bytes);
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }

        pack_buffer = ptr;
        pack_write_offset = (GLintptr)((size_t)offset + pack_layout.skip_offset_bytes);
        pack_write_size = (GLsizeiptr)pack_layout.write_span_bytes;
        pixels = (void *)(base + offset + pack_layout.skip_offset_bytes);
    }

    if (!STATE(buffers[_PIXEL_PACK_BUFFER]) && pixels == NULL)
    {
        fprintf(stderr, "MGL Error: mglReadPixels: pixels is NULL with no pixel pack buffer bound\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (!STATE(buffers[_PIXEL_PACK_BUFFER])) {
        pixels = (void *)((uint8_t *)pixels + pack_layout.skip_offset_bytes);
    }

    if (format == GL_DEPTH_COMPONENT)
    {
        if (!mglReadPixelsDepthComponent(ctx, pixels, &pack_layout, pack_buffer,
                                         pack_write_offset, pack_write_size,
                                         x, y, width, height, format, type))
            return;
        return;
    }

    if ((format == GL_RED_INTEGER || format == GL_RG_INTEGER ||
         format == GL_RGB_INTEGER || format == GL_BGR_INTEGER ||
         format == GL_RGBA_INTEGER || format == GL_BGRA_INTEGER ||
         format == 0x8d95 /*GL_GREEN_INTEGER*/ ||
         format == 0x8d96 /*GL_BLUE_INTEGER*/ ||
         format == 0x8d97 /*GL_ALPHA_INTEGER*/) &&
        (type == GL_BYTE || type == GL_UNSIGNED_BYTE ||
         type == GL_SHORT || type == GL_UNSIGNED_SHORT ||
         type == GL_INT || type == GL_UNSIGNED_INT ||
         type == GL_UNSIGNED_BYTE_3_3_2 || type == GL_UNSIGNED_BYTE_2_3_3_REV ||
         type == GL_UNSIGNED_SHORT_5_6_5 || type == GL_UNSIGNED_SHORT_5_6_5_REV ||
         type == GL_UNSIGNED_SHORT_4_4_4_4 || type == GL_UNSIGNED_SHORT_4_4_4_4_REV ||
         type == GL_UNSIGNED_SHORT_5_5_5_1 || type == GL_UNSIGNED_SHORT_1_5_5_5_REV ||
         type == GL_UNSIGNED_INT_8_8_8_8 || type == GL_UNSIGNED_INT_8_8_8_8_REV ||
         type == GL_UNSIGNED_INT_10_10_10_2 || type == GL_UNSIGNED_INT_2_10_10_10_REV))
    {
        if (pack_layout.dst_pitch > UINT_MAX ||
            pack_layout.write_span_bytes > UINT_MAX)
        {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }

        mglFlushCommandBuffer(ctx);
        mglRendererReadIntegerPixels(ctx,
                                            pixels,
                                            (GLuint)pack_layout.dst_pitch,
                                            (GLuint)pack_layout.write_span_bytes,
                                            x,
                                            y,
                                            width,
                                            height,
                                            format,
                                            type);
        if (STATE(pack.swap_bytes) == GL_TRUE) {
            size_t elem_size = mglPixelTypeDatumBytes(type);
            if (elem_size > 1u) {
                mglSwapReadPixelsOutput((uint8_t *)pixels, pack_layout.write_span_bytes, elem_size);
            }
        }
        if (pack_buffer)
            mglMarkPackBufferReadPixelsWrite(ctx, pack_buffer, pack_write_offset, pack_write_size, pixels);
        return;
    }

    if (format == GL_STENCIL_INDEX)
    {
        if (!mglReadPixelsStencilIndex(ctx, pixels, &pack_layout, pack_buffer,
                                       pack_write_offset, pack_write_size,
                                       x, y, width, height, type))
            return;
        return;
    }

    if (format == GL_DEPTH_STENCIL)
    {
        if (!mglReadPixelsDepthStencil(ctx, pixels, &pack_layout, pack_buffer,
                                       pack_write_offset, pack_write_size,
                                       x, y, width, height, type))
            return;
        return;
    }

    if (STATE(read_buffer) == GL_NONE)
    {
        fprintf(stderr, "MGL Error: mglReadPixels: read buffer is GL_NONE\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    FBOAttachment *readColorAttachment = NULL;
    Texture *readColorTexture = NULL;
    if (ctx->state.readbuffer &&
        ctx->state.read_buffer >= GL_COLOR_ATTACHMENT0 &&
        ctx->state.read_buffer < GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS) {
        readColorAttachment =
            &ctx->state.readbuffer->color_attachments[ctx->state.read_buffer - GL_COLOR_ATTACHMENT0];
        readColorTexture = mglStencilAttachmentTexture(readColorAttachment);
    }
    if (readColorTexture &&
        readColorTexture->rgb10a2_shadow &&
        (readColorTexture->internalformat == GL_RGB10 ||
         readColorTexture->internalformat == GL_RGB10_A2 ||
         readColorTexture->internalformat == GL_RGB10_A2UI)) {
        size_t shadowPitch = (size_t)readColorTexture->rgb10a2_shadow_width * 4u;
        size_t stagingPitch = (size_t)width * 4u;
        uint8_t *staging = calloc((size_t)height, stagingPitch);
        if (!staging) {
            ERROR_RETURN(GL_OUT_OF_MEMORY);
            return;
        }
        for (GLsizei row = 0; row < height; row++) {
            GLint readY = y + row;
            if (readY < 0 || readY >= (GLint)readColorTexture->rgb10a2_shadow_height) continue;
            for (GLsizei column = 0; column < width; column++) {
                GLint readX = x + column;
                if (readX < 0 || readX >= (GLint)readColorTexture->rgb10a2_shadow_width) continue;
                memcpy(staging + ((size_t)row * width + column) * 4u,
                       readColorTexture->rgb10a2_shadow + (size_t)readY * shadowPitch + (size_t)readX * 4u,
                       4u);
            }
        }
        GLboolean packed = mglPackBGRA8ReadPixels(staging, stagingPitch, pixels,
                                                  pack_layout.dst_pitch, width, height, format, type);
        free(staging);
        if (!packed) {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
        if (STATE(pack.swap_bytes) == GL_TRUE) {
            size_t elem_size = mglPixelTypeDatumBytes(type);
            if (elem_size > 1u) {
                mglSwapReadPixelsOutput((uint8_t *)pixels, pack_layout.write_span_bytes, elem_size);
            }
        }
        if (pack_buffer)
            mglMarkPackBufferReadPixelsWrite(ctx, pack_buffer, pack_write_offset, pack_write_size, pixels);
        return;
    }

    /* Integer texture formats need to be read back via mtlGetTexImage (which
     * has a dedicated integer readback path) rather than via mtlReadDrawable
     * (which converts to BGRA8 UNORM, losing integer semantics). */
    if (readColorTexture && mglIsIntegerReadFormat(format))
    {
        GLuint level = readColorAttachment ? readColorAttachment->level : 0u;
        GLuint slice = (readColorAttachment && !readColorAttachment->layered)
            ? readColorAttachment->layer
            : 0u;

        if (pack_layout.dst_pitch > UINT_MAX ||
            pack_layout.write_span_bytes > UINT_MAX)
        {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }

        mglFlushCommandBuffer(ctx);
        mglRendererGetTexImage(ctx,
                                      readColorTexture,
                                      pixels,
                                      (GLuint)pack_layout.dst_pitch,
                                      (GLuint)pack_layout.write_span_bytes,
                                      x,
                                      y,
                                      width,
                                      height,
                                      format,
                                      type,
                                      level,
                                      slice);
        if (STATE(pack.swap_bytes) == GL_TRUE) {
            size_t elem_size = mglPixelTypeDatumBytes(type);
            if (elem_size > 1u) {
                mglSwapReadPixelsOutput((uint8_t *)pixels, pack_layout.write_span_bytes, elem_size);
            }
        }
        if (pack_buffer)
            mglMarkPackBufferReadPixelsWrite(ctx, pack_buffer, pack_write_offset, pack_write_size, pixels);
        return;
    }

    /* SNORM and other non-BGRA8 texture formats attached to FBOs need to be
     * read back via mtlGetTexImage (which supports format conversion) rather
     * than via mtlReadDrawable (which only returns BGRA8 from the drawable). */
    if (readColorTexture &&
        readColorTexture->internalformat != GL_RGBA8 &&
        readColorTexture->internalformat != GL_RGB8 &&
        readColorTexture->internalformat != GL_SRGB8 &&
        readColorTexture->internalformat != GL_SRGB8_ALPHA8 &&
        readColorTexture->internalformat != GL_R8 &&
        readColorTexture->internalformat != GL_RG8 &&
        mglIsColorReadFormat(format))
    {
        GLuint level = readColorAttachment ? readColorAttachment->level : 0u;
        GLuint slice = (readColorAttachment && !readColorAttachment->layered)
            ? readColorAttachment->layer
            : 0u;

        if (pack_layout.dst_pitch > UINT_MAX ||
            pack_layout.write_span_bytes > UINT_MAX)
        {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }

        mglFlushCommandBuffer(ctx);
        mglRendererGetTexImage(ctx,
                                      readColorTexture,
                                      pixels,
                                      (GLuint)pack_layout.dst_pitch,
                                      (GLuint)pack_layout.write_span_bytes,
                                      x,
                                      y,
                                      width,
                                      height,
                                      format,
                                      type,
                                      level,
                                      slice);
        if (STATE(pack.swap_bytes) == GL_TRUE) {
            size_t elem_size = mglPixelTypeDatumBytes(type);
            if (elem_size > 1u) {
                mglSwapReadPixelsOutput((uint8_t *)pixels, pack_layout.write_span_bytes, elem_size);
            }
        }
        if (pack_buffer)
            mglMarkPackBufferReadPixelsWrite(ctx, pack_buffer, pack_write_offset, pack_write_size, pixels);
        return;
    }

    size_t readback_pitch = 0u;
    size_t readback_size = 0u;
    if (!mglMulSizeT((size_t)width, 4u, &readback_pitch) ||
        !mglMulSizeT(readback_pitch, (size_t)height, &readback_size) ||
        readback_size > UINT_MAX)
    {
        fprintf(stderr,
                "MGL Error: mglReadPixels: readback staging overflow width=%d height=%d\n",
                width,
                height);
        ERROR_RETURN(GL_OUT_OF_MEMORY);
        return;
    }

    mglFlushCommandBuffer(ctx);

    kern_return_t err;
    vm_address_t buffer_data;

    err = vm_allocate((vm_map_t) mach_task_self(),
                      (vm_address_t*) &buffer_data,
                      (vm_size_t)readback_size,
                      VM_FLAGS_ANYWHERE);
    if (err)
    {
        ERROR_RETURN(GL_OUT_OF_MEMORY);
        return;
    }

    mglRendererReadDrawable(ctx, (void *)buffer_data, (GLuint)readback_pitch, (GLuint)readback_size, x, y, width, height);


    if (!mglPackBGRA8ReadPixels((const uint8_t *)buffer_data,
                                readback_pitch,
                                (uint8_t *)pixels,
                                pack_layout.dst_pitch,
                                width,
                                height,
                                format,
                                type))
    {
        static uint64_t s_unsupported_readpixels_pack_count = 0u;
        uint64_t hit = ++s_unsupported_readpixels_pack_count;
        if (hit <= 32u || (hit % 256u) == 0u) {
            fprintf(stderr,
                    "MGL WARNING: mglReadPixels unsupported BGRA8 pack conversion format=0x%x type=0x%x hit=%llu\n",
                    format,
                    type,
                    (unsigned long long)hit);
        }
        ERROR_RETURN(GL_INVALID_OPERATION);
        vm_deallocate(mach_task_self(), buffer_data, readback_size);
        return;
    }

    if (STATE(pack.swap_bytes) == GL_TRUE) {
        size_t elem_size = mglPixelTypeDatumBytes(type);
        if (elem_size > 1u) {
            mglSwapReadPixelsOutput((uint8_t *)pixels, pack_layout.write_span_bytes, elem_size);
        }
    }

    if (pack_buffer)
        mglMarkPackBufferReadPixelsWrite(ctx, pack_buffer, pack_write_offset, pack_write_size, pixels);

    vm_deallocate(mach_task_self(), buffer_data, readback_size);
}
