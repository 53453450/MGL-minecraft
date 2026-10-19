/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_debug_fbo_stats.c -- MGL_DEBUG_FBO_STATS=<frame>
 *
 * Diagnostic only.  During the swap interval numbered <frame> (the value of
 * the SWAP_ENTRY call counter), every time the draw framebuffer changes, the
 * outgoing framebuffer's color attachments are sampled at a few points and
 * the result is printed to stderr as FBOSTAT lines, and the default
 * framebuffer is sampled right before the swap.  This finds the first pass
 * whose output is already empty.
 */

#include "mgl_debug_fbo_stats.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "glm_context.h"
#include "mgl_renderer_ports.h"
#include "draw_command.h"

extern void mglBindFramebuffer(GLMContext ctx, GLenum target, GLuint framebuffer);
extern void mglReadBuffer(GLMContext ctx, GLenum buf);
extern GLenum mglGetError(GLMContext ctx);
extern void mglReadPixels(GLMContext ctx, GLint x, GLint y, GLsizei width,
                          GLsizei height, GLenum format, GLenum type, void *pixels);

static uint64_t s_frame = 0;
static int s_target_frame = -2;   /* -2: not read yet, -1: disabled */
static int s_busy = 0;

static int mglDebugFboStatsActive(void)
{
    if (s_target_frame == -2) {
        const char *v = getenv("MGL_DEBUG_FBO_STATS");
        s_target_frame = (v && v[0]) ? atoi(v) : -1;
    }
    return s_target_frame >= 0 && (uint64_t)s_target_frame == s_frame && !s_busy;
}

void mglDebugFboStatsSetFrame(uint64_t swap_call)
{
    s_frame = swap_call;
}

static void mglDebugSampleBound(GLMContext ctx, GLenum read_buf, GLuint width,
                                GLuint height, const char *label, GLuint fbo,
                                int att, GLuint tex, GLenum ifmt)
{
    mglReadBuffer(ctx, read_buf);
    float sum[4] = {0, 0, 0, 0}, mx[4] = {0, 0, 0, 0};
    int n = 0;
    static const float fx[5] = {0.5f, 0.2f, 0.8f, 0.2f, 0.8f};
    static const float fy[5] = {0.5f, 0.2f, 0.2f, 0.8f, 0.8f};
    for (int i = 0; i < 5; i++) {
        float px[4] = {0, 0, 0, 0};
        GLint x = (GLint)(fx[i] * (float)(width ? width - 1 : 0));
        GLint y = (GLint)(fy[i] * (float)(height ? height - 1 : 0));
        mglReadPixels(ctx, x, y, 1, 1, GL_RGBA, GL_FLOAT, px);
        for (int c = 0; c < 4; c++) {
            sum[c] += px[c];
            if (px[c] > mx[c]) mx[c] = px[c];
        }
        n++;
    }
    fprintf(stderr,
            "FBOSTAT frame=%llu %s fbo=%u att=%d tex=%u ifmt=0x%x size=%ux%u "
            "mean=(%.4f %.4f %.4f %.4f) max=(%.4f %.4f %.4f %.4f)\n",
            (unsigned long long)s_frame, label, fbo, att, tex, ifmt, width,
            height, sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n, mx[0],
            mx[1], mx[2], mx[3]);
}

void mglDebugFboStatsOnDrawFramebufferChange(GLMContext ctx, Framebuffer *old_fbo)
{
    if (!ctx || !old_fbo || !mglDebugFboStatsActive()) return;
    s_busy = 1;
    mglFlushPendingDraws(ctx);
    Framebuffer *saved_read = ctx->state.readbuffer;
    GLuint saved_read_name = saved_read ? saved_read->name : 0u;
    GLenum saved_read_buf = ctx->state.read_buffer;
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (((old_fbo->color_attachment_bitfield >> i) & 1u) == 0u) continue;
        FBOAttachment *att = &old_fbo->color_attachments[i];
        Texture *t = mglRendererAttachmentTextureFor(ctx, att);
        mglBindFramebuffer(ctx, GL_READ_FRAMEBUFFER, old_fbo->name);
        mglDebugSampleBound(ctx, GL_COLOR_ATTACHMENT0 + (GLenum)i,
                            t ? t->width : 0u, t ? t->height : 0u, "pass-end",
                            old_fbo->name, i, att->texture,
                            t ? (GLenum)t->internalformat : 0u);
    }
    mglBindFramebuffer(ctx, GL_READ_FRAMEBUFFER, saved_read_name);
    mglReadBuffer(ctx, saved_read_buf);
    while (mglGetError(ctx) != GL_NO_ERROR) {
    }
    s_busy = 0;
}

void mglDebugFboStatsAtSwap(GLMContext ctx, GLuint width, GLuint height)
{
    if (!ctx || !mglDebugFboStatsActive()) return;
    s_busy = 1;
    mglFlushPendingDraws(ctx);
    Framebuffer *saved_read = ctx->state.readbuffer;
    GLuint saved_read_name = saved_read ? saved_read->name : 0u;
    GLenum saved_read_buf = ctx->state.read_buffer;
    mglBindFramebuffer(ctx, GL_READ_FRAMEBUFFER, 0);
    mglDebugSampleBound(ctx, GL_BACK, width, height, "window-before-swap", 0, 0,
                        0, 0);
    mglBindFramebuffer(ctx, GL_READ_FRAMEBUFFER, saved_read_name);
    mglReadBuffer(ctx, saved_read_buf);
    while (mglGetError(ctx) != GL_NO_ERROR) {
    }
    s_busy = 0;
}
