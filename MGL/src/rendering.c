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
#include <string.h>

#include "mgl.h"

#include "pixel_utils.h"
#include "glm_context.h"
#include "mgl_safety.h"

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
    uint64_t callCount = ++s_mglClearCallCount;

    if (mask & ~(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT))
    {
        fprintf(stderr, "MGL Error: mglClear: invalid mask 0x%x\n", mask);
        ERROR_RETURN(GL_INVALID_VALUE);
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
        fprintf(stderr,
                "MGL TRACE clear.set call=%llu mask=0x%x prevMask=0x%x drawBuf=0x%x readBuf=0x%x fbo=%p(%u) dirty=0x%x\n",
                (unsigned long long)callCount,
                (unsigned)mask,
                (unsigned)previousMask,
                (unsigned)ctx->state.draw_buffer,
                (unsigned)ctx->state.read_buffer,
                (void *)ctx->state.framebuffer,
                (unsigned)mglSafeDrawFramebufferName(ctx),
                (unsigned)ctx->state.dirty_bits);
    }
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
        fprintf(stderr,
                "MGL TRACE clearColor call=%llu value=(%.3f,%.3f,%.3f,%.3f) fbo=%p(%u) drawBuf=0x%x dirty=0x%x\n",
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
            break;
    }

    ctx->state.dirty_bits |= DIRTY_FBO | DIRTY_STATE;

    if (mglShouldTraceClearCall(callCount)) {
        fprintf(stderr,
                "MGL TRACE clearBufferfv call=%llu buffer=0x%x drawbuffer=%d fbo=%p(%u) value=(%.3f,%.3f,%.3f,%.3f)\n",
                (unsigned long long)callCount,
                (unsigned)buffer,
                (int)drawbuffer,
                (void *)fbo,
                (unsigned)(fbo ? fbo->name : 0),
                value ? value[0] : 0.0f,
                value ? value[1] : 0.0f,
                value ? value[2] : 0.0f,
                value ? value[3] : 0.0f);
    }
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
            break;
    }

    ctx->state.dirty_bits |= DIRTY_FBO | DIRTY_STATE;

    if (mglShouldTraceClearCall(callCount)) {
        fprintf(stderr,
                "MGL TRACE clearBufferfi call=%llu buffer=0x%x drawbuffer=%d fbo=%p(%u) depth=%.3f stencil=%d\n",
                (unsigned long long)callCount,
                (unsigned)buffer,
                (int)drawbuffer,
                (void *)fbo,
                (unsigned)(fbo ? fbo->name : 0),
                depth,
                stencil);
    }
}

void mglFinish(GLMContext ctx)
{
    fprintf(stderr, "MGL: mglFinish called - flushing and waiting for GPU\n");
    ctx->mtl_funcs.mtlFlush(ctx, true);
}

void mglFlush(GLMContext ctx)
{
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

    if ((STATE(draw_buffer) != buf ||
         STATE(draw_buffer_count) != 1 ||
         STATE(draw_buffers[0]) != buf) &&
        ctx->mtl_funcs.mtlInvalidateRenderPass)
    {
        ctx->mtl_funcs.mtlInvalidateRenderPass(ctx);
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

    STATE(read_buffer) = buf;
    mglStoreCurrentReadBufferSelection(ctx);
    STATE(dirty_bits) |= DIRTY_FBO | DIRTY_STATE;
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

    if (type != GL_UNSIGNED_BYTE && type != GL_UNSIGNED_INT_8_8_8_8_REV)
        return false;

    for (GLsizei y = 0; y < height; y++)
    {
        const uint8_t *src_row = src + ((size_t)y * src_pitch);
        uint8_t *dst_row = dst + ((size_t)y * dst_pitch);

        switch(format)
        {
            case GL_BGRA:
                memcpy(dst_row, src_row, (size_t)width * 4u);
                break;

            case GL_RGBA:
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
            ERROR_CHECK_RETURN(ctx->stencil_format.mtl_pixel_format > 0, GL_INVALID_OPERATION);
            break;

        case GL_DEPTH_COMPONENT:
            ERROR_CHECK_RETURN(ctx->depth_format.mtl_pixel_format > 0, GL_INVALID_OPERATION);
            break;

        case GL_DEPTH_STENCIL:
            ERROR_CHECK_RETURN((ctx->depth_format.mtl_pixel_format > 0) ||
                               (ctx->stencil_format.mtl_pixel_format > 0), GL_INVALID_OPERATION);
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

        // ERROR_CHECK_RETURN(ptr->mapped == false, GL_INVALID_OPERATION);
        if (ptr->mapped) {
            fprintf(stderr, "MGL Error: mglReadPixels: pixel pack buffer is mapped\n");
            ERROR_RETURN(GL_INVALID_OPERATION);
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

    if (STATE(read_buffer) == GL_NONE)
    {
        fprintf(stderr, "MGL Error: mglReadPixels: read buffer is GL_NONE\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    if (format == GL_DEPTH_COMPONENT ||
        format == GL_STENCIL_INDEX ||
        format == GL_DEPTH_STENCIL)
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

    vm_deallocate(mach_task_self(), buffer_data, readback_size);
}
