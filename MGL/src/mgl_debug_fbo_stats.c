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
#include "mgl_renderer_attachment.h"
#include "mgl_renderer_state_areas.h"
#include "draw_command.h"

extern void mglBindFramebuffer(GLMContext ctx, GLenum target, GLuint framebuffer);
extern void mglReadBuffer(GLMContext ctx, GLenum buf);
extern GLenum mglGetError(GLMContext ctx);
extern GLint mglGetUniformLocation(GLMContext ctx, GLuint program, const GLchar *name);
extern void mglGetUniformfv(GLMContext ctx, GLuint program, GLint location, GLfloat *params);

extern void mglGetProgramiv(GLMContext ctx, GLuint program, GLenum pname, GLint *params);
extern void mglGetActiveUniform(GLMContext ctx, GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name);
extern void mglGetUniformiv(GLMContext ctx, GLuint program, GLint location, GLint *params);

/* Write every active uniform of `program` to /tmp/mgl_uniforms_prog<N>.txt as
 * "name type count v0 v1 ..." so a standalone harness can replay the draw. */
static void mglDebugDumpAllUniforms(GLMContext ctx, GLuint program)
{
    char path[128];
    snprintf(path, sizeof(path), "/tmp/mgl_uniforms_prog%u.txt", (unsigned)program);
    FILE *f = fopen(path, "w");
    GLint count = 0;
    mglGetProgramiv(ctx, program, GL_ACTIVE_UNIFORMS, &count);
    {
        FILE *lg = fopen("/tmp/mgl_fbostat.log", "a");
        if (lg) {
            fprintf(lg, "FBOSTAT-UNIFORM-DUMP program=%u active=%d file=%s opened=%d\n",
                    (unsigned)program, (int)count, path, f ? 1 : 0);
            fclose(lg);
        }
    }
    if (!f) return;
    for (GLint i = 0; i < count; i++) {
        char name[256] = {0};
        GLsizei len = 0;
        GLint size = 0;
        GLenum type = 0;
        mglGetActiveUniform(ctx, program, (GLuint)i, sizeof(name), &len, &size, &type, name);
        for (GLint e = 0; e < size; e++) {
            char ename[300];
            if (size > 1) {
                char base[256];
                snprintf(base, sizeof(base), "%s", name);
                char *br = strchr(base, '[');
                if (br) *br = 0;
                snprintf(ename, sizeof(ename), "%s[%d]", base, e);
            } else {
                snprintf(ename, sizeof(ename), "%s", name);
            }
            GLint loc = mglGetUniformLocation(ctx, program, ename);
            if (loc < 0) continue;
            int is_int = (type == 0x1404 || type == 0x1405 || (type >= 0x8B53 && type <= 0x8B59) ||
                          (type >= 0x8B5E && type <= 0x8B62) || type == 0x8DC1 || (type >= 0x8DC0 && type <= 0x8DD7) );
            fprintf(f, "%s %u ", ename, (unsigned)type);
            if (is_int) {
                GLint v[16] = {0};
                mglGetUniformiv(ctx, program, loc, v);
                fprintf(f, "i 4");
                for (int k = 0; k < 4; k++) fprintf(f, " %d", v[k]);
            } else {
                float v[16] = {0};
                mglGetUniformfv(ctx, program, loc, v);
                fprintf(f, "f 16");
                for (int k = 0; k < 16; k++) fprintf(f, " %.9g", v[k]);
            }
            fprintf(f, "\n");
        }
    }
    fclose(f);
}

/* Print a few uniforms of the program that drew last in the pass. */
static void mglDebugDumpUniforms(GLMContext ctx, GLuint program)
{
    static const struct { const char *name; int n; } names[] = {
        {"viewWidth", 1}, {"viewHeight", 1}, {"near", 1}, {"far", 1},
        {"gbufferProjectionInverse", 16}, {"upVec", 3}, {"sunVec", 3},
        {"frameTimeCounter", 1}, {"rainStrength", 1}, {"sunAngle", 1},
    };
    FILE *outs[2] = {stderr, NULL};
    outs[1] = fopen(getenv("MGL_DEBUG_FBO_STATS_FILE") ? getenv("MGL_DEBUG_FBO_STATS_FILE") : "/tmp/mgl_fbostat.log", "a");
    for (int o = 0; o < 2; o++) {
        if (!outs[o]) continue;
        fprintf(outs[o], "FBOSTAT-UNIFORMS program=%u", (unsigned)program);
        for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
            GLint loc = mglGetUniformLocation(ctx, program, names[i].name);
            fprintf(outs[o], " %s[loc=%d]=", names[i].name, loc);
            if (loc < 0) { fprintf(outs[o], "-"); continue; }
            float v[16] = {0};
            mglGetUniformfv(ctx, program, loc, v);
            fprintf(outs[o], "(");
            for (int k = 0; k < names[i].n; k++) fprintf(outs[o], k ? ",%g" : "%g", v[k]);
            fprintf(outs[o], ")");
        }
        fprintf(outs[o], "\n");
        if (outs[o] != stderr) fclose(outs[o]);
    }
}
extern void mglReadPixels(GLMContext ctx, GLint x, GLint y, GLsizei width,
                          GLsizei height, GLenum format, GLenum type, void *pixels);

static uint64_t s_frame = 0;
static int s_target_frame = -2;   /* -2: not read yet, -1: disabled */
static int s_busy = 0;

static FILE *mglDebugFboStatsFile(void)
{
    static FILE *f = NULL;
    static int tried = 0;
    if (!tried) {
        tried = 1;
        const char *path = getenv("MGL_DEBUG_FBO_STATS_FILE");
        f = fopen(path && path[0] ? path : "/tmp/mgl_fbostat.log", "a");
    }
    return f;
}

static int mglDebugFboStatsActive(void)
{
    if (s_target_frame == -2) {
        /* The frame number comes from MGL_DEBUG_FBO_STATS, or from the first
         * line of /tmp/mgl_fbostat.arm when the launcher's environment cannot
         * be changed. */
        const char *v = getenv("MGL_DEBUG_FBO_STATS");
        if (v && v[0]) {
            s_target_frame = atoi(v);
        } else {
            FILE *arm = fopen("/tmp/mgl_fbostat.arm", "r");
            int n = -1;
            if (arm) {
                if (fscanf(arm, "%d", &n) != 1) n = -1;
                fclose(arm);
            }
            s_target_frame = n;
        }
    }
    return s_target_frame >= 0 && (uint64_t)s_target_frame == s_frame && !s_busy;
}

void mglDebugFboStatsSetFrame(uint64_t swap_call)
{
    s_frame = swap_call;
}

static int mglDebugSampleBound(GLMContext ctx, GLenum read_buf, GLuint width,
                                GLuint height, const char *label, GLuint fbo,
                                int att, GLuint tex, GLenum ifmt)
{
    mglReadBuffer(ctx, read_buf);
    float sum[4] = {0, 0, 0, 0}, mx[4] = {0, 0, 0, 0};
    int n = 0, nans = 0;
    static const float fx[5] = {0.5f, 0.2f, 0.8f, 0.2f, 0.8f};
    static const float fy[5] = {0.5f, 0.2f, 0.2f, 0.8f, 0.8f};
    for (int i = 0; i < 5; i++) {
        float px[4] = {0, 0, 0, 0};
        GLint x = (GLint)(fx[i] * (float)(width ? width - 1 : 0));
        GLint y = (GLint)(fy[i] * (float)(height ? height - 1 : 0));
        mglReadPixels(ctx, x, y, 1, 1, GL_RGBA, GL_FLOAT, px);
        for (int c = 0; c < 4; c++) {
            if (px[c] != px[c]) {   /* NaN */
                nans++;
                continue;
            }
            sum[c] += px[c];
            if (px[c] > mx[c]) mx[c] = px[c];
        }
        n++;
    }
    FILE *outs[2] = {stderr, mglDebugFboStatsFile()};
    for (int o = 0; o < 2; o++) {
        if (!outs[o]) continue;
        fprintf(outs[o],
                "FBOSTAT frame=%llu %s fbo=%u att=%d tex=%u ifmt=0x%x size=%ux%u "
                "mean=(%.4f %.4f %.4f %.4f) max=(%.4f %.4f %.4f %.4f) nan_components=%d program=%u\n",
                (unsigned long long)s_frame, label, fbo, att, tex, ifmt, width,
                height, sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n, mx[0],
                mx[1], mx[2], mx[3], nans, (unsigned)ctx->state.program_name);
        if (outs[o] != stderr) fflush(outs[o]);
    }
    return nans;
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
        if (i == 0) mglDebugDumpUniforms(ctx, ctx->state.program_name);
        int nan_count = mglDebugSampleBound(ctx, GL_COLOR_ATTACHMENT0 + (GLenum)i,
                            t ? t->width : 0u, t ? t->height : 0u, "pass-end",
                            old_fbo->name, i, att->texture,
                            t ? (GLenum)t->internalformat : 0u);
        static GLuint s_dumped_program = 0;
        if ((nan_count > 0 || getenv("MGL_DEBUG_FBO_STATS_ALWAYS_DUMP")) && ctx->state.program_name != 0u &&
            s_dumped_program == 0u) {
            s_dumped_program = ctx->state.program_name;
            mglDebugDumpAllUniforms(ctx, ctx->state.program_name);
        }
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
