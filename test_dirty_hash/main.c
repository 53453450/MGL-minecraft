/*
 * Minimal regression for dirty-hash invalidation across a deferred flush.
 *
 * The failure sequence is:
 *   draw(A) -> state(B) -> glFlush() -> draw(B) -> state(A) -> draw(A)
 *
 * A flush must not discard the hash invalidation for state B. If it does,
 * the final two draws receive the same MGLStateKey and are merged into the
 * first draw's B-state snapshot.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GL_GLEXT_PROTOTYPES 1
#include <GL/glcorearb.h>

#include "draw_command.h"
#include "glm_context.h"
#include "MGLRenderer.h"

#define TEST_W 32
#define TEST_H 32

static GLuint compile_shader(GLenum type, const char *source)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        char log[2048] = {0};
        glGetShaderInfoLog(shader, sizeof(log), NULL, log);
        fprintf(stderr, "dirty-hash: shader compile failed: %s\n", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static GLuint make_program(void)
{
    static const char *vertex_source =
        "#version 330 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *fragment_source =
        "#version 330 core\n"
        "out vec4 color;\n"
        "void main() { color = vec4(1.0, 0.0, 0.0, 1.0); }\n";

    GLuint vertex = compile_shader(GL_VERTEX_SHADER, vertex_source);
    GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    if (!vertex || !fragment) {
        if (vertex) glDeleteShader(vertex);
        if (fragment) glDeleteShader(fragment);
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[2048] = {0};
        glGetProgramInfoLog(program, sizeof(log), NULL, log);
        fprintf(stderr, "dirty-hash: program link failed: %s\n", log);
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

static GLuint make_vao(const GLfloat vertices[6], GLuint *out_buffer)
{
    GLuint vao = 0;
    GLuint buffer = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(GLfloat), vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    *out_buffer = buffer;
    return vao;
}

static GLuint make_fbo(GLuint *color_texture)
{
    GLuint fbo = 0;
    GLuint texture = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, TEST_W, TEST_H, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, texture, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "dirty-hash: framebuffer is incomplete\n");
        return 0;
    }
    *color_texture = texture;
    return fbo;
}

static int verify_stable_hash_cache(GLMContext ctx)
{
    const GLuint domain_bits = MGL_TEXTURE_HASH_DIRTY_BITS |
                               MGL_VERTEX_LAYOUT_HASH_DIRTY_BITS |
                               MGL_RENDER_STATE_HASH_DIRTY_BITS;
    const GLuint saved_dirty_bits = ctx->active_state->dirty_bits;
    const uint8_t saved_texture_dirty = ctx->active_state->texture_dirty;
    const uint8_t saved_vertex_dirty = ctx->active_state->vertex_layout_dirty;
    const uint8_t saved_render_dirty = ctx->active_state->render_state_dirty;
    const uint64_t original_texture = ctx->active_state->cached_texture_hash;
    const uint64_t original_vertex = ctx->active_state->cached_vertex_layout_hash;
    const uint64_t original_render = ctx->active_state->cached_render_state_hash;
    MGLStateKey first_key;
    MGLStateKey second_key;

    mglMarkStateDirtyBits(ctx->active_state, domain_bits);
    mglComputeStateKey(ctx, GL_TRIANGLES, false, &first_key);

    if (ctx->active_state->texture_dirty ||
        ctx->active_state->vertex_layout_dirty ||
        ctx->active_state->render_state_dirty) {
        fprintf(stderr, "dirty-hash: initial key did not consume cache flags\n");
        return 1;
    }
    if ((ctx->active_state->dirty_bits & domain_bits) != domain_bits) {
        fprintf(stderr, "dirty-hash: key computation consumed renderer dirty bits\n");
        return 1;
    }

    const uint64_t computed_texture = ctx->active_state->cached_texture_hash;
    const uint64_t computed_vertex = ctx->active_state->cached_vertex_layout_hash;
    const uint64_t computed_render = ctx->active_state->cached_render_state_hash;
    const uint64_t texture_canary = computed_texture ^ 0x6a09e667f3bcc909ULL;
    const uint64_t vertex_canary = computed_vertex ^ 0xbb67ae8584caa73bULL;
    const uint64_t render_canary = computed_render ^ 0x3c6ef372fe94f82bULL;

    ctx->active_state->cached_texture_hash = texture_canary;
    ctx->active_state->cached_vertex_layout_hash = vertex_canary;
    ctx->active_state->cached_render_state_hash = render_canary;
    mglComputeStateKey(ctx, GL_TRIANGLES, false, &second_key);

    if (ctx->active_state->cached_texture_hash != texture_canary ||
        ctx->active_state->cached_vertex_layout_hash != vertex_canary ||
        ctx->active_state->cached_render_state_hash != render_canary) {
        fprintf(stderr, "dirty-hash: stable key recomputed from legacy dirty bits\n");
        return 1;
    }
    if (ctx->active_state->texture_dirty ||
        ctx->active_state->vertex_layout_dirty ||
        ctx->active_state->render_state_dirty) {
        fprintf(stderr, "dirty-hash: stable key unexpectedly dirtied a cache\n");
        return 1;
    }
    if ((ctx->active_state->dirty_bits & domain_bits) != domain_bits) {
        fprintf(stderr, "dirty-hash: stable key consumed renderer dirty bits\n");
        return 1;
    }

    mglClearStateDirtyBitsPreservingHashInvalidation(ctx->active_state);
    if (ctx->active_state->texture_dirty ||
        ctx->active_state->vertex_layout_dirty ||
        ctx->active_state->render_state_dirty) {
        fprintf(stderr, "dirty-hash: renderer-bit clear re-invalidated stable caches\n");
        return 1;
    }

    ctx->active_state->dirty_bits = saved_dirty_bits;
    ctx->active_state->texture_dirty = saved_texture_dirty;
    ctx->active_state->vertex_layout_dirty = saved_vertex_dirty;
    ctx->active_state->render_state_dirty = saved_render_dirty;
    ctx->active_state->cached_texture_hash = original_texture;
    ctx->active_state->cached_vertex_layout_hash = original_vertex;
    ctx->active_state->cached_render_state_hash = original_render;
    return 0;
}

static int verify_transform_feedback_binding_hash(GLMContext ctx)
{
    GLuint buffer = 0;
    MGLStateKey unbound_key;
    MGLStateKey bound_key;
    MGLStateKey restored_key;

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0);
    mglComputeStateKey(ctx, GL_TRIANGLES, false, &unbound_key);

    glGenBuffers(1, &buffer);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buffer);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 64, NULL, GL_STATIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buffer);

    BufferBaseTarget *slot =
        &ctx->active_state->buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[0];
    if (!buffer || slot->buffer != buffer || !slot->buf) {
        fprintf(stderr, "dirty-hash: transform-feedback base bind failed\n");
        return 1;
    }

    mglComputeStateKey(ctx, GL_TRIANGLES, false, &bound_key);
    if (bound_key.render_state_hash == unbound_key.render_state_hash) {
        fprintf(stderr, "dirty-hash: transform-feedback binding missing from state key\n");
        return 1;
    }

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0);
    mglComputeStateKey(ctx, GL_TRIANGLES, false, &restored_key);
    if (restored_key.render_state_hash != unbound_key.render_state_hash) {
        fprintf(stderr, "dirty-hash: transform-feedback unbind did not restore state key\n");
        return 1;
    }

    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
    glDeleteBuffers(1, &buffer);
    return 0;
}

int main(void)
{
    static const GLfloat offscreen_triangle[6] = {
        2.0f, 2.0f, 3.0f, 2.0f, 2.5f, 3.0f,
    };
    static const GLfloat visible_triangle[6] = {
        -0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f,
    };

    setenv("MGL_OPT_DIRTY_HASH", "1", 1);
    if (!getenv("MGL_DIRTY_KEY_DELTA"))
        setenv("MGL_DIRTY_KEY_DELTA", "0", 1);
    if (!getenv("MGL_PARALLEL_ENCODE"))
        setenv("MGL_PARALLEL_ENCODE", "0", 1);
    unsetenv("MGL_DISABLE_DRAW_DEFER");

    GLMContext ctx = createGLMContext(
        GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV,
        GL_DEPTH_COMPONENT, GL_FLOAT, 0, 0);
    if (!ctx || !CppCreateMGLRendererHeadless(ctx)) {
        fprintf(stderr, "dirty-hash: failed to create headless context\n");
        return 1;
    }
    MGLsetCurrentContext(ctx);

    GLuint color_texture = 0;
    GLuint fbo = make_fbo(&color_texture);
    GLuint program = make_program();
    GLuint offscreen_buffer = 0;
    GLuint visible_buffer = 0;
    GLuint offscreen_vao = make_vao(offscreen_triangle, &offscreen_buffer);
    GLuint visible_vao = make_vao(visible_triangle, &visible_buffer);
    if (!fbo || !program || !offscreen_vao || !visible_vao) {
        return 1;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, TEST_W, TEST_H);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(program);
    glEnable(GL_CULL_FACE);
    if (verify_stable_hash_cache(ctx) != 0) {
        return 1;
    }
    if (verify_transform_feedback_binding_hash(ctx) != 0) {
        return 1;
    }

    /* Cache state A (GL_BACK) in a real deferred batch. The offscreen
       geometry keeps this warm-up draw out of the pixel assertion. */
    glBindVertexArray(offscreen_vao);
    glCullFace(GL_BACK);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    /* Mutate to B without computing a new draw key, then flush A. */
    glCullFace(GL_FRONT);
    glFlush();
    if (ctx->draw_command_buffer.batch_count != 0) {
        fprintf(stderr, "dirty-hash: warm-up flush left pending batches\n");
        return 1;
    }

    /* B is culled. A is visible. They must remain separate batches. */
    glBindVertexArray(visible_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glCullFace(GL_BACK);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    uint32_t batch_count = ctx->draw_command_buffer.batch_count;
    uint32_t command_count = ctx->draw_command_buffer.total_commands;
    if (batch_count != 2 || command_count != 2) {
        fprintf(stderr,
                "dirty-hash: expected 2 batches/2 commands, got %u/%u\n",
                batch_count,
                command_count);
        return 1;
    }

    glFinish();

    GLubyte center[4] = {0, 0, 0, 0};
    glReadPixels(TEST_W / 2, TEST_H / 2, 1, 1,
                 GL_RGBA, GL_UNSIGNED_BYTE, center);
    if (center[0] < 200 || center[1] > 32 || center[2] > 32) {
        fprintf(stderr,
                "dirty-hash: visible A draw was lost, center=%u,%u,%u,%u\n",
                center[0], center[1], center[2], center[3]);
        return 1;
    }

    glUseProgram(0);
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteProgram(program);
    glDeleteVertexArrays(1, &offscreen_vao);
    glDeleteVertexArrays(1, &visible_vao);
    glDeleteBuffers(1, &offscreen_buffer);
    glDeleteBuffers(1, &visible_buffer);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &color_texture);
    glFinish();

    printf("dirty-hash batch regression: PASS\n");
    return 0;
}
