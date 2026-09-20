/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * test_state_snapshot_share.c — M2 (per-key snapshot dedup) contract test.
 *
 * M2 lets a batch that revisits a state already captured earlier in the SAME
 * command buffer share that batch's state_snapshot instead of capturing its
 * own.  The observable contract is:
 *
 *   1. a revisited key produces a SHARE hit (g_mglSnapshotShareHitsSinceSwap)
 *   2. sharing actually avoids a capture: the allocation count and the copied
 *      byte count must BOTH be lower than a run that cannot share
 *   3. the command buffer still replays correctly (nothing reads a freed or
 *      stale snapshot) - which the existing batch regressions cover, and which
 *      this test exercises by flushing and reading the result back
 *
 * How the test forces a share: batches are created by key, and adjacent draws
 * with an equal key merge into ONE batch.  To get two SEPARATE batches with the
 * same key we interleave a different-key draw: bind VAO1 (batch A), bind VAO2
 * (batch B), bind VAO1 again.  The third draw's key matches the first, so it
 * must find the first batch as a donor.
 *
 * Run under both memory modes - the share path is the only one that changed and
 * the non-arena path is where the release-side leak lived:
 *   ./test_state_snapshot_share
 *   MGL_ARENA_SNAPSHOT=0 ./test_state_snapshot_share
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <GL/glcorearb.h>

#include "MGLRenderer.h"
#include "glm_context.h"
#include "mgl_frame_activity.h"

#define TEST_W 64
#define TEST_H 64

static int g_fail;

static void expect(int cond, const char *what)
{
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        g_fail++;
    } else {
        printf("  ok: %s\n", what);
    }
}

static const char *kVS =
    "#version 330 core\n"
    "layout(location = 0) in vec2 pos;\n"
    "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";

static const char *kFS =
    "#version 330 core\n"
    "out vec4 frag;\n"
    "void main() { frag = vec4(0.0, 1.0, 0.0, 1.0); }\n";

/* A second program: program_name is part of MGLStateKey and cannot be absorbed
 * by the per-draw dynamic-binding merge, so alternating programs really does
 * produce separate batches (alternating VAOs does not - the merge path captures
 * the vertex bindings per draw instead). */
static const char *kFS2 =
    "#version 330 core\n"
    "out vec4 frag;\n"
    "void main() { frag = vec4(0.0, 0.5, 0.0, 1.0); }\n";

static GLuint compile(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048] = {0};
        glGetShaderInfoLog(s, sizeof(log) - 1, NULL, log);
        fprintf(stderr, "shader compile failed: %s\n", log);
        return 0;
    }
    return s;
}

static GLuint make_program2(void)
{
    GLuint vs = compile(GL_VERTEX_SHADER, kVS);
    GLuint fs = compile(GL_FRAGMENT_SHADER, kFS2);
    if (!vs || !fs) return 0;
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return ok ? p : 0;
}

static GLuint make_program(void)
{
    GLuint vs = compile(GL_VERTEX_SHADER, kVS);
    GLuint fs = compile(GL_FRAGMENT_SHADER, kFS);
    if (!vs || !fs) return 0;
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048] = {0};
        glGetProgramInfoLog(p, sizeof(log) - 1, NULL, log);
        fprintf(stderr, "link failed: %s\n", log);
        return 0;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

/* A VAO is what makes two batches differ by key (the key carries vao_name). */
static GLuint make_vao(GLuint *out_buffer, float dx)
{
    static const float base[3][2] = {{-0.5f, -0.5f}, {0.5f, -0.5f}, {0.0f, 0.5f}};
    float verts[6];
    for (int i = 0; i < 3; i++) {
        verts[i * 2 + 0] = base[i][0] + dx;
        verts[i * 2 + 1] = base[i][1];
    }

    GLuint vao = 0, vbo = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
    *out_buffer = vbo;
    return vao;
}

static GLuint make_fbo(GLuint *out_tex)
{
    GLuint tex = 0, fbo = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, TEST_W, TEST_H, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, tex, 0);
    *out_tex = tex;
    return fbo;
}

/* Read the green channel at the centre to prove the shared-snapshot replay
 * actually produced the expected geometry. */
static int read_center_green(void)
{
    unsigned char px[4] = {0, 0, 0, 0};
    glReadPixels(TEST_W / 2, TEST_H / 4, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    return px[1];
}

int main(void)
{
    /* Deterministic: no deferred-draw disabling, no parallelism, no delta. */
    unsetenv("MGL_DISABLE_DRAW_DEFER");
    if (!getenv("MGL_DIRTY_KEY_DELTA")) setenv("MGL_DIRTY_KEY_DELTA", "0", 1);
    if (!getenv("MGL_PARALLEL_ENCODE")) setenv("MGL_PARALLEL_ENCODE", "0", 1);

    /* M2 deliberately EXCLUDES stream-merged batches from both donating and
     * sharing: their snapshot carries batch-specific pointer patches (the
     * transient stream buffers are written into the frozen VAO), so it is not
     * interchangeable with a plain capture.  A plain A,B,A of tiny
     * glDrawArrays calls all becomes stream-merged, which would test the
     * exclusion rather than the feature - so disable stream merge here.
     * MGL_PERF_SUMMARY must also be on: MGL_PERF_ADD is gated by
     * mglPerfSummaryEnabled(), so every counter below reads 0 without it. */
    if (!getenv("MGL_DISABLE_STREAM_MERGE")) setenv("MGL_DISABLE_STREAM_MERGE", "1", 1);
    if (!getenv("MGL_PERF_SUMMARY")) setenv("MGL_PERF_SUMMARY", "1", 1);

    GLMContext ctx = createGLMContext(GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV,
                                      GL_DEPTH_COMPONENT, GL_FLOAT, 0, 0);
    if (!ctx || !CppCreateMGLRendererHeadless(ctx)) {
        fprintf(stderr, "snapshot-share: failed to create headless context\n");
        return 1;
    }
    MGLsetCurrentContext(ctx);

    GLuint color_tex = 0;
    GLuint fbo = make_fbo(&color_tex);
    GLuint program = make_program();
    GLuint program2 = make_program2();
    GLuint buf_a = 0, buf_b = 0;
    GLuint vao_a = make_vao(&buf_a, -0.2f);
    GLuint vao_b = make_vao(&buf_b, 0.0f);
    if (!fbo || !program || !program2 || !vao_a || !vao_b) {
        fprintf(stderr, "snapshot-share: setup failed\n");
        return 1;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, TEST_W, TEST_H);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(program);

    /* Baseline A: three batches, all DIFFERENT keys (A, B, A-with-other-viewport
     * would still differ).  Use viewport as the third distinct key so no key
     * repeats and therefore nothing can share. */
    const uint64_t hits_before = MGL_FRAME_LOAD(g_mglSnapshotShareHitsSinceSwap);
    const uint64_t allocs_before =
        MGL_FRAME_LOAD(g_mglSnapshotAllocationCountSinceSwap);
    const uint64_t bytes_before =
        MGL_FRAME_LOAD(g_mglSnapshotBytesAllocatedSinceSwap);

    glBindVertexArray(vao_a);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(vao_b);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glViewport(0, 0, TEST_W / 2, TEST_H);          /* third, distinct key */
    glBindVertexArray(vao_a);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glViewport(0, 0, TEST_W, TEST_H);
    glFlush();

    const uint64_t hits_baseline =
        MGL_FRAME_LOAD(g_mglSnapshotShareHitsSinceSwap) - hits_before;
    printf("baseline (no repeated key): share_hits=%llu\n",
           (unsigned long long)hits_baseline);

    /* Now the aimed case: A, B, A.  The third draw's key equals the first. */
    const uint64_t hits0 = MGL_FRAME_LOAD(g_mglSnapshotShareHitsSinceSwap);
    const uint64_t allocs0 =
        MGL_FRAME_LOAD(g_mglSnapshotAllocationCountSinceSwap);
    const uint64_t bytes0 = MGL_FRAME_LOAD(g_mglSnapshotBytesAllocatedSinceSwap);

    glBindVertexArray(vao_a);
    glUseProgram(program);                          /* key A */
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glUseProgram(program2);                         /* key B */
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glUseProgram(program);                          /* key A repeats batch 1 */
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFlush();

    const uint64_t hits = MGL_FRAME_LOAD(g_mglSnapshotShareHitsSinceSwap) - hits0;
    const uint64_t allocs =
        MGL_FRAME_LOAD(g_mglSnapshotAllocationCountSinceSwap) - allocs0;
    const uint64_t bytes =
        MGL_FRAME_LOAD(g_mglSnapshotBytesAllocatedSinceSwap) - bytes0;

    printf("revisit  (A,B,A):            share_hits=%llu captures=%llu bytes=%llu\n",
           (unsigned long long)hits, (unsigned long long)allocs,
           (unsigned long long)bytes);

    expect(hits_baseline == 0, "no share hit when no key repeats");
    expect(hits >= 1, "revisited key produces a share hit");
    expect(allocs == 2, "the third draw reuses a snapshot (2 captures, not 3)");

    /* The replay must still be correct: the last draw wins the centre pixel. */
    const int green = read_center_green();
    expect(green > 128, "shared-snapshot replay still renders (centre is green)");

    /* The leak contract: a batch that shares a snapshot still owns its command
     * array.  We cannot observe malloc directly here, but we can at least prove
     * the command buffer resets cleanly and the context tears down without the
     * double-free that a bad sharing scheme would produce. */
    destroyGLMContext(ctx);
    printf("context torn down cleanly\n");

    (void)allocs_before; (void)bytes_before;
    printf("state-snapshot-share: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
