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

extern void mglTraceLogExternal(const char *fmt, ...);

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
    if (!texture || !mglEnsureDepthShadow(texture)) return;
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
    if (!source || !destination || !source->depth_shadow || !mglEnsureDepthShadow(destination) ||
        srcX1 <= srcX0 || srcY1 <= srcY0 || dstX1 <= dstX0 || dstY1 <= dstY0) return;
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
    if (!source || !source->rgb10a2_shadow) return;

    GLsizei drawBufferCount = ctx->state.draw_buffer_count;
    if (drawBufferCount > MAX_COLOR_ATTACHMENTS) drawBufferCount = MAX_COLOR_ATTACHMENTS;
    for (GLsizei slot = 0; slot < drawBufferCount; slot++) {
        GLenum drawBuffer = ctx->state.draw_buffers[slot];
        if (drawBuffer < GL_COLOR_ATTACHMENT0 ||
            drawBuffer >= GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS) continue;
        Texture *destination =
            mglStencilAttachmentTexture(&drawFBO->color_attachments[drawBuffer - GL_COLOR_ATTACHMENT0]);
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
        ctx->state.dirty_bits |= DIRTY_BUFFER;
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
        !ctx->mtl_funcs.mtlClearBuffer ||
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

    ctx->mtl_funcs.mtlClearBuffer(ctx, 0, mask);
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
        if (ctx->mtl_funcs.mtlClearBuffer) {
            ctx->mtl_funcs.mtlClearBuffer(ctx, 0, mask);
        }
        if ((mask & GL_STENCIL_BUFFER_BIT) &&
            ctx->state.framebuffer &&
            (ctx->state.var.stencil_writemask != 0u ||
             ctx->state.var.stencil_back_writemask != 0u)) {
            ctx->state.framebuffer->stencil.clear_color[0] =
                (GLfloat)ctx->state.var.stencil_clear_value;
        }
        ctx->state.clear_bitmask = 0;
        ctx->state.dirty_bits |= DIRTY_FBO | DIRTY_STATE | DIRTY_RENDER_STATE;
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
    ctx->state.dirty_bits |= DIRTY_FBO | DIRTY_STATE;

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

    ctx->state.dirty_bits |= DIRTY_STATE;

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

    ctx->state.dirty_bits |= DIRTY_STATE;
}

void mglClearDepth(GLMContext ctx, GLdouble depth)
{
    ctx->state.var.depth_clear_value = mglClampDepthClearValue(depth);

    ctx->state.dirty_bits |= DIRTY_STATE;

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
            break;
        default:
            fprintf(stderr, "MGL Error: mglClearBufferfv: invalid buffer 0x%x\n", buffer);
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    ctx->state.dirty_bits |= DIRTY_FBO | DIRTY_STATE;

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
            break;
        default:
            fprintf(stderr, "MGL Error: mglClearBufferfi: invalid buffer 0x%x\n", buffer);
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    ctx->state.dirty_bits |= DIRTY_FBO | DIRTY_STATE;

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
    fprintf(stderr, "MGL: mglFinish called - flushing and waiting for GPU\n");
    mglFlushCommandBuffer(ctx);
    ctx->mtl_funcs.mtlFlush(ctx, true);
}

void mglFlush(GLMContext ctx)
{
    mglFlushCommandBuffer(ctx);
    ctx->mtl_funcs.mtlFlush(ctx, false);
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

    if (changed && ctx->mtl_funcs.mtlInvalidateRenderPass)
        ctx->mtl_funcs.mtlInvalidateRenderPass(ctx);

    for (GLsizei i = 0; i < n; ++i)
        STATE(draw_buffers[i]) = bufs[i];
    for (GLsizei i = n; i < (GLsizei)MAX_COLOR_ATTACHMENTS; ++i)
        STATE(draw_buffers[i]) = GL_NONE;

    STATE(draw_buffer_count) = n;
    STATE(draw_buffer) = (n > 0) ? bufs[0] : GL_NONE;
    mglStoreCurrentDrawBufferSelection(ctx);
    STATE(dirty_bits) |= DIRTY_FBO | DIRTY_STATE | DIRTY_RENDER_STATE;
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

        if (changed && ctx->mtl_funcs.mtlInvalidateRenderPass)
            ctx->mtl_funcs.mtlInvalidateRenderPass(ctx);

        STATE(draw_buffer) = GL_NONE;
        STATE(draw_buffer_count) = 1;
        STATE(draw_buffers[0]) = GL_NONE;
        for (GLuint i = 1; i < MAX_COLOR_ATTACHMENTS; ++i)
            STATE(draw_buffers[i]) = GL_NONE;
        mglStoreCurrentDrawBufferSelection(ctx);
        STATE(dirty_bits) |= DIRTY_FBO | DIRTY_STATE;
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

    if (changed && ctx->mtl_funcs.mtlInvalidateRenderPass)
        ctx->mtl_funcs.mtlInvalidateRenderPass(ctx);

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
    STATE(dirty_bits) |= DIRTY_FBO | DIRTY_STATE;
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
        fprintf(stderr, "MGL Error: mglReadBuffer: non-attachment buffer 0x%x is invalid for user FBO\n", buf);
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
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

    if (type != GL_UNSIGNED_BYTE &&
        type != GL_UNSIGNED_INT_8_8_8_8_REV &&
        type != GL_UNSIGNED_SHORT_4_4_4_4 &&
        type != GL_UNSIGNED_SHORT_5_5_5_1 &&
        type != GL_FLOAT)
        return false;

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
                    default:
                        return false;
                }
            }
            continue;
        }

        switch(format)
        {
            case GL_RGBA:
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
                for (GLsizei x = 0; x < width; x++)
                {
                    const uint8_t *s = src_row + ((size_t)x * 4u);
                    uint8_t *d = dst_row + ((size_t)x * 2u);
                    d[0] = s[2];
                    d[1] = s[1];
                }
                break;

            case GL_RED:
                for (GLsizei x = 0; x < width; x++)
                {
                    dst_row[x] = src_row[((size_t)x * 4u) + 2u];
                }
                break;

            default:
                return false;
        }
    }

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
    }

    // ERROR_CHECK_RETURN(width > 0, GL_INVALID_ENUM);
    if (width < 0) {
        fprintf(stderr, "MGL Error: mglReadPixels: width < 0 (%d)\n", width);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    // ERROR_CHECK_RETURN(height > 0, GL_INVALID_ENUM);
    if (height < 0) {
        fprintf(stderr, "MGL Error: mglReadPixels: height < 0 (%d)\n", height);
        ERROR_RETURN(GL_INVALID_VALUE);
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
            }
            break;

        case GL_DEPTH_COMPONENT:
            if (!ctx->state.readbuffer) {
                ERROR_CHECK_RETURN(ctx->depth_format.mtl_pixel_format > 0, GL_INVALID_OPERATION);
            }
            break;

        case GL_DEPTH_STENCIL:
            if (!ctx->state.readbuffer) {
                ERROR_CHECK_RETURN((ctx->depth_format.mtl_pixel_format > 0) ||
                                   (ctx->stencil_format.mtl_pixel_format > 0), GL_INVALID_OPERATION);
            }
            switch(type)
            {
                case GL_UNSIGNED_INT_24_8:
                case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
                    break;

                default:
                    ERROR_RETURN(GL_INVALID_ENUM);
                    break;
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
            // ERROR_CHECK_RETURN(format == GL_RGB || format == GL_BGR, GL_INVALID_OPERATION);
            if (!(format == GL_RGB || format == GL_BGR)) {
                fprintf(stderr, "MGL Error: mglReadPixels: invalid format for type (format=0x%x type=0x%x)\n", format, type);
                ERROR_RETURN(GL_INVALID_OPERATION);
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
            // ERROR_CHECK_RETURN(format == GL_RGBA || format == GL_BGRA, GL_INVALID_OPERATION);
            if (!(format == GL_RGBA || format == GL_BGRA)) {
                fprintf(stderr, "MGL Error: mglReadPixels: invalid format for type (format=0x%x type=0x%x)\n", format, type);
                ERROR_RETURN(GL_INVALID_OPERATION);
            }
            break;
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
        }

        offset = (uintptr_t)pixels;
        if (pixel_size && (offset % (uintptr_t)pixel_size) != 0)
        {
            fprintf(stderr, "MGL Error: mglReadPixels: pixel pack buffer offset not aligned (offset=%lu pixel_size=%u)\n", (unsigned long)offset, pixel_size);
            ERROR_RETURN(GL_INVALID_OPERATION);
        }

        if ((size_t)ptr->size < offset || (size_t)ptr->size - offset < pack_layout.required_bytes)
        {
            fprintf(stderr, "MGL Error: mglReadPixels: pixel pack buffer too small (size=%ld offset=%lu req=%zu)\n",
                    (long)ptr->size, (unsigned long)offset, pack_layout.required_bytes);
            ERROR_RETURN(GL_INVALID_OPERATION);
        }

        base = (uint8_t *)(uintptr_t)ptr->data.buffer_data;
        if (!base)
        {
            fprintf(stderr, "MGL Error: mglReadPixels: pixel pack buffer has no CPU storage\n");
            ERROR_RETURN(GL_INVALID_OPERATION);
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
    }

    if (!STATE(buffers[_PIXEL_PACK_BUFFER])) {
        pixels = (void *)((uint8_t *)pixels + pack_layout.skip_offset_bytes);
    }

    if (format == GL_DEPTH_COMPONENT)
    {
        Texture *depthTexture = ctx->state.readbuffer
            ? mglStencilAttachmentTexture(&ctx->state.readbuffer->depth)
            : NULL;
        if (type == GL_FLOAT && depthTexture && depthTexture->depth_shadow) {
            for (GLsizei row = 0; row < height; row++) {
                GLfloat *dst = (GLfloat *)((uint8_t *)pixels + (size_t)row * pack_layout.dst_pitch);
                for (GLsizei column = 0; column < width; column++) {
                    GLint readX = x + column;
                    GLint readY = y + row;
                    dst[column] = (readX >= 0 && readY >= 0 &&
                                   readX < (GLint)depthTexture->depth_shadow_width &&
                                   readY < (GLint)depthTexture->depth_shadow_height)
                        ? depthTexture->depth_shadow[(size_t)readY * depthTexture->depth_shadow_width + readX]
                        : 0.0f;
                }
            }
            if (pack_buffer)
                mglMarkPackBufferReadPixelsWrite(ctx, pack_buffer, pack_write_offset, pack_write_size, pixels);
            return;
        }
        if (type != GL_FLOAT || !ctx->mtl_funcs.mtlReadDepthPixels)
        {
            static uint64_t s_unsupported_depth_readpixels_count = 0u;
            uint64_t hit = ++s_unsupported_depth_readpixels_count;
            if (hit <= 32u || (hit % 256u) == 0u) {
                fprintf(stderr,
                        "MGL WARNING: mglReadPixels depth readback is unsupported format=0x%x type=0x%x hit=%llu\n",
                        format,
                        type,
                        (unsigned long long)hit);
            }
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }

        if (pack_layout.dst_pitch > UINT_MAX ||
            pack_layout.write_span_bytes > UINT_MAX)
        {
            fprintf(stderr,
                    "MGL Error: mglReadPixels: depth readback pack overflow pitch=%zu span=%zu\n",
                    pack_layout.dst_pitch,
                    pack_layout.write_span_bytes);
            ERROR_RETURN(GL_OUT_OF_MEMORY);
            return;
        }

        mglFlushCommandBuffer(ctx);
        ctx->mtl_funcs.mtlReadDepthPixels(ctx,
                                          pixels,
                                          (GLuint)pack_layout.dst_pitch,
                                          (GLuint)pack_layout.write_span_bytes,
                                          x,
                                          y,
                                          width,
                                          height);
        if (pack_buffer)
            mglMarkPackBufferReadPixelsWrite(ctx, pack_buffer, pack_write_offset, pack_write_size, pixels);
        return;
    }

    if ((format == GL_RED_INTEGER || format == GL_RG_INTEGER ||
         format == GL_RGB_INTEGER || format == GL_RGBA_INTEGER) &&
        (type == GL_BYTE || type == GL_UNSIGNED_BYTE ||
         type == GL_SHORT || type == GL_UNSIGNED_SHORT ||
         type == GL_INT || type == GL_UNSIGNED_INT))
    {
        if (!ctx->mtl_funcs.mtlReadIntegerPixels ||
            pack_layout.dst_pitch > UINT_MAX ||
            pack_layout.write_span_bytes > UINT_MAX)
        {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }

        mglFlushCommandBuffer(ctx);
        ctx->mtl_funcs.mtlReadIntegerPixels(ctx,
                                            pixels,
                                            (GLuint)pack_layout.dst_pitch,
                                            (GLuint)pack_layout.write_span_bytes,
                                            x,
                                            y,
                                            width,
                                            height,
                                            format,
                                            type);
        if (pack_buffer)
            mglMarkPackBufferReadPixelsWrite(ctx, pack_buffer, pack_write_offset, pack_write_size, pixels);
        return;
    }

    if (format == GL_STENCIL_INDEX)
    {
        if (type != GL_UNSIGNED_BYTE && type != GL_UNSIGNED_INT && type != GL_INT)
        {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
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
            uint8_t *dst = (uint8_t *)pixels + (size_t)row * pack_layout.dst_pitch;
            if (type == GL_UNSIGNED_BYTE) {
                for (GLsizei column = 0; column < width; column++) {
                    GLint readX = x + column;
                    GLint readY = y + row;
                    dst[column] = (stencilTexture && stencilTexture->stencil_shadow &&
                                   readX >= 0 && readY >= 0 &&
                                   readX < (GLint)stencilTexture->stencil_shadow_width &&
                                   readY < (GLint)stencilTexture->stencil_shadow_height)
                        ? stencilTexture->stencil_shadow[(size_t)readY * stencilTexture->stencil_shadow_width + readX]
                        : value;
                }
            } else {
                GLuint *dst32 = (GLuint *)(void *)dst;
                for (GLsizei column = 0; column < width; column++) {
                    GLint readX = x + column;
                    GLint readY = y + row;
                    GLuint stencilValue = (stencilTexture && stencilTexture->stencil_shadow &&
                                     readX >= 0 && readY >= 0 &&
                                     readX < (GLint)stencilTexture->stencil_shadow_width &&
                                     readY < (GLint)stencilTexture->stencil_shadow_height)
                        ? stencilTexture->stencil_shadow[(size_t)readY * stencilTexture->stencil_shadow_width + readX]
                        : value;
                    if (type == GL_INT)
                        ((GLint *)(void *)dst32)[column] = (GLint)stencilValue;
                    else
                        dst32[column] = stencilValue;
                }
            }
        }
        if (pack_buffer)
            mglMarkPackBufferReadPixelsWrite(ctx, pack_buffer, pack_write_offset, pack_write_size, pixels);
        return;
    }

    if (format == GL_DEPTH_STENCIL)
    {
        static uint64_t s_unsupported_depth_stencil_readpixels_count = 0u;
        uint64_t hit = ++s_unsupported_depth_stencil_readpixels_count;
        if (hit <= 32u || (hit % 256u) == 0u) {
            fprintf(stderr,
                    "MGL WARNING: mglReadPixels depth/stencil readback is unsupported format=0x%x type=0x%x hit=%llu\n",
                    format,
                    type,
                    (unsigned long long)hit);
        }
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (STATE(read_buffer) == GL_NONE)
    {
        fprintf(stderr, "MGL Error: mglReadPixels: read buffer is GL_NONE\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    Texture *readColorTexture = NULL;
    if (ctx->state.readbuffer &&
        ctx->state.read_buffer >= GL_COLOR_ATTACHMENT0 &&
        ctx->state.read_buffer < GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS) {
        readColorTexture = mglStencilAttachmentTexture(
            &ctx->state.readbuffer->color_attachments[ctx->state.read_buffer - GL_COLOR_ATTACHMENT0]);
    }
    if (readColorTexture && readColorTexture->rgb10a2_shadow) {
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
    }

    ctx->mtl_funcs.mtlReadDrawable(ctx, (void *)buffer_data, (GLuint)readback_pitch, (GLuint)readback_size, x, y, width, height);

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
    }

    if (pack_buffer)
        mglMarkPackBufferReadPixelsWrite(ctx, pack_buffer, pack_write_offset, pack_write_size, pixels);

    vm_deallocate(mach_task_self(), buffer_data, readback_size);
}
