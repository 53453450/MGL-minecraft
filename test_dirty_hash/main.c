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

static GLuint make_color_fbo_size(GLsizei width, GLsizei height,
                                  GLuint *color_texture)
{
    GLuint fbo = 0;
    GLuint texture = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
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

static GLuint make_fbo(GLuint *color_texture)
{
    return make_color_fbo_size(TEST_W, TEST_H, color_texture);
}

static GLuint make_depth_fbo_size(GLsizei width, GLsizei height,
                                  GLuint *depth_texture)
{
    GLuint fbo = 0;
    GLuint texture = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F,
                 width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                           GL_TEXTURE_2D, texture, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "dirty-hash: depth framebuffer is incomplete\n");
        return 0;
    }
    *depth_texture = texture;
    return fbo;
}

static int verify_air_aux_render_pipelines(void)
{
    GLuint color_fbos[2] = {0u, 0u};
    GLuint color_textures[2] = {0u, 0u};
    GLuint depth_fbos[2] = {0u, 0u};
    GLuint depth_textures[2] = {0u, 0u};
    GLint saved_read_fbo = 0;
    GLint saved_draw_fbo = 0;
    GLint saved_viewport[4] = {0, 0, 0, 0};
    GLint saved_scissor_box[4] = {0, 0, 0, 0};
    GLfloat saved_clear_color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    GLfloat saved_clear_depth = 1.0f;
    GLboolean saved_scissor = glIsEnabled(GL_SCISSOR_TEST);
    GLubyte center[4] = {0u, 0u, 0u, 0u};
    GLubyte corner[4] = {0u, 0u, 0u, 0u};
    GLfloat depth = 1.0f;
    int failed = 1;

    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &saved_read_fbo);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &saved_draw_fbo);
    glGetIntegerv(GL_VIEWPORT, saved_viewport);
    glGetIntegerv(GL_SCISSOR_BOX, saved_scissor_box);
    glGetFloatv(GL_COLOR_CLEAR_VALUE, saved_clear_color);
    glGetFloatv(GL_DEPTH_CLEAR_VALUE, &saved_clear_depth);
    while (glGetError() != GL_NO_ERROR) {}

    color_fbos[0] = make_color_fbo_size(4, 4, &color_textures[0]);
    color_fbos[1] = make_color_fbo_size(8, 8, &color_textures[1]);
    if (!color_fbos[0] || !color_fbos[1]) goto done;

    glDisable(GL_SCISSOR_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, color_fbos[0]);
    glViewport(0, 0, 4, 4);
    glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, color_fbos[1]);
    glViewport(0, 0, 8, 8);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, color_fbos[0]);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, color_fbos[1]);
    glBlitFramebuffer(0, 0, 4, 4, 0, 0, 8, 8,
                      GL_COLOR_BUFFER_BIT, GL_LINEAR);
    glBlitFramebuffer(0, 0, 4, 4, 0, 0, 8, 8,
                      GL_COLOR_BUFFER_BIT, GL_LINEAR);
    glFinish();

    glBindFramebuffer(GL_READ_FRAMEBUFFER, color_fbos[1]);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(4, 4, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, center);
    if (center[0] < 48u || center[0] > 80u ||
        center[1] < 112u || center[1] > 144u ||
        center[2] < 176u || center[2] > 208u) {
        fprintf(stderr,
                "dirty-hash: scaled color blit mismatch=%u,%u,%u,%u\n",
                center[0], center[1], center[2], center[3]);
        goto done;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, color_fbos[1]);
    glEnable(GL_SCISSOR_TEST);
    glScissor(2, 2, 4, 4);
    glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_COLOR_BUFFER_BIT);
    glFinish();
    glDisable(GL_SCISSOR_TEST);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(4, 4, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, center);
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, corner);
    if (center[0] > 32u || center[1] < 220u || center[2] > 32u ||
        corner[0] < 48u || corner[1] < 112u || corner[2] < 176u) {
        fprintf(stderr,
                "dirty-hash: scissored clear mismatch center=%u,%u,%u,%u corner=%u,%u,%u,%u\n",
                center[0], center[1], center[2], center[3],
                corner[0], corner[1], corner[2], corner[3]);
        goto done;
    }

    depth_fbos[0] = make_depth_fbo_size(4, 4, &depth_textures[0]);
    depth_fbos[1] = make_depth_fbo_size(8, 8, &depth_textures[1]);
    if (!depth_fbos[0] || !depth_fbos[1]) goto done;

    glBindFramebuffer(GL_FRAMEBUFFER, depth_fbos[0]);
    glViewport(0, 0, 4, 4);
    glClearDepth(0.25);
    glClear(GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, depth_fbos[1]);
    glViewport(0, 0, 8, 8);
    glClearDepth(1.0);
    glClear(GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, depth_fbos[0]);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, depth_fbos[1]);
    glBlitFramebuffer(0, 0, 4, 4, 0, 0, 8, 8,
                      GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    glBlitFramebuffer(0, 0, 4, 4, 0, 0, 8, 8,
                      GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    glFinish();
    glBindFramebuffer(GL_READ_FRAMEBUFFER, depth_fbos[1]);
    glReadPixels(4, 4, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
    if (depth < 0.20f || depth > 0.30f || glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "dirty-hash: scaled depth blit mismatch=%f\n", depth);
        goto done;
    }

    printf("AIR_AUX_RENDER_PSO_OK color=%u,%u,%u,%u depth=%.3f\n",
           center[0], center[1], center[2], center[3], depth);
    failed = 0;

done:
    if (saved_scissor) glEnable(GL_SCISSOR_TEST);
    else glDisable(GL_SCISSOR_TEST);
    glScissor(saved_scissor_box[0], saved_scissor_box[1],
              saved_scissor_box[2], saved_scissor_box[3]);
    glClearColor(saved_clear_color[0], saved_clear_color[1],
                 saved_clear_color[2], saved_clear_color[3]);
    glClearDepth(saved_clear_depth);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint)saved_read_fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, (GLuint)saved_draw_fbo);
    glViewport(saved_viewport[0], saved_viewport[1],
               saved_viewport[2], saved_viewport[3]);
    glDeleteFramebuffers(2, color_fbos);
    glDeleteTextures(2, color_textures);
    glDeleteFramebuffers(2, depth_fbos);
    glDeleteTextures(2, depth_textures);
    return failed;
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

static int verify_buffer_range_lifecycle(GLMContext ctx)
{
    enum { binding_index = 3 };
    GLuint buffers[2] = {0, 0};
    GLint alignment = 1;
    GLint generic_binding = 0;
    GLint64 indexed_binding = 0;
    GLint64 indexed_start = 0;
    GLint64 indexed_size = 0;

    while (glGetError() != GL_NO_ERROR) {}
    glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &alignment);
    if (alignment < 1) alignment = 1;

    glGenBuffers(2, buffers);
    glBindBuffer(GL_UNIFORM_BUFFER, buffers[0]);
    glBufferData(GL_UNIFORM_BUFFER, 64, NULL, GL_DYNAMIC_DRAW);
    glBindBufferRange(GL_UNIFORM_BUFFER, binding_index, buffers[0], 0, 64);
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "dirty-hash: failed to establish range canary\n");
        return 1;
    }

    /* Error paths must not replace either the indexed or generic binding. */
    glBindBufferRange(GL_UNIFORM_BUFFER, binding_index, buffers[1],
                      -(GLintptr)alignment, 16);
    if (glGetError() != GL_INVALID_VALUE) {
        fprintf(stderr, "dirty-hash: negative range offset was accepted\n");
        return 1;
    }
    glBindBufferRange(GL_UNIFORM_BUFFER, binding_index, buffers[1], 0, 0);
    if (glGetError() != GL_INVALID_VALUE) {
        fprintf(stderr, "dirty-hash: zero range size was accepted\n");
        return 1;
    }
    glBindBufferRange(GL_ARRAY_BUFFER, binding_index, buffers[1], 0, 16);
    if (glGetError() != GL_INVALID_ENUM) {
        fprintf(stderr, "dirty-hash: invalid range target was accepted\n");
        return 1;
    }
    glBindBufferBase(GL_ARRAY_BUFFER, binding_index, buffers[1]);
    if (glGetError() != GL_INVALID_ENUM) {
        fprintf(stderr, "dirty-hash: invalid base target was accepted\n");
        return 1;
    }

    glGetInteger64i_v(GL_UNIFORM_BUFFER_BINDING, binding_index, &indexed_binding);
    glGetInteger64i_v(GL_UNIFORM_BUFFER_START, binding_index, &indexed_start);
    glGetInteger64i_v(GL_UNIFORM_BUFFER_SIZE, binding_index, &indexed_size);
    glGetIntegerv(GL_UNIFORM_BUFFER_BINDING, &generic_binding);
    if ((GLuint)indexed_binding != buffers[0] || indexed_start != 0 ||
        indexed_size != 64 || (GLuint)generic_binding != buffers[0]) {
        fprintf(stderr, "dirty-hash: invalid range call mutated binding state\n");
        return 1;
    }

    /* GL 4.2+ permits a range beyond the current store.  Its visible size is
       clamped only when consumed, including after later storage resizes. */
    glBindBufferRange(GL_UNIFORM_BUFFER, binding_index, buffers[1], 0, 8192);
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "dirty-hash: pre-storage oversized range was rejected\n");
        return 1;
    }

    BufferBaseTarget *slot =
        &ctx->active_state->buffer_base[_UNIFORM_BUFFER].buffers[binding_index];
    BufferMap map = {0};
    map.buf = slot->buf;
    map.offset = slot->offset;
    map.size = slot->size;
    if (slot->buffer != buffers[1] || slot->size != 8192 ||
        mglBufferMapVisibleSize(&map) != 0) {
        fprintf(stderr, "dirty-hash: pre-storage range state is inconsistent\n");
        return 1;
    }

    glBufferData(GL_UNIFORM_BUFFER, 64, NULL, GL_DYNAMIC_DRAW);
    if (glGetError() != GL_NO_ERROR || mglBufferMapVisibleSize(&map) != 64) {
        fprintf(stderr, "dirty-hash: range did not clamp to initial store\n");
        return 1;
    }
    glBufferData(GL_UNIFORM_BUFFER, 16, NULL, GL_DYNAMIC_DRAW);
    if (glGetError() != GL_NO_ERROR || mglBufferMapVisibleSize(&map) != 16) {
        fprintf(stderr, "dirty-hash: range did not follow smaller store\n");
        return 1;
    }

    map.size = 8;
    if (mglBufferMapStorageRemaining(&map) != 16 ||
        mglBufferMapVisibleSize(&map) != 8 ||
        mglBufferMapAvailableBackingBytes(&map, 64) != 16 ||
        mglBufferMapVisibleBackingBytes(&map, 64) != 8) {
        fprintf(stderr, "dirty-hash: storage and declared range were conflated\n");
        return 1;
    }

    map.offset = 32;
    map.size = 8192;
    if (mglBufferMapAvailableBackingBytes(&map, 64) != 0 ||
        mglBufferMapVisibleBackingBytes(&map, 64) != 0) {
        fprintf(stderr, "dirty-hash: backing bytes escaped logical store clamp\n");
        return 1;
    }
    map.offset = 0;

    /* The same oversized range remains legal when a store already exists. */
    glBindBufferRange(GL_UNIFORM_BUFFER, binding_index, buffers[1], 0, 16384);
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "dirty-hash: allocated oversized range was rejected\n");
        return 1;
    }
    map.size = 16384;
    if (mglBufferMapVisibleSize(&map) != 16) {
        fprintf(stderr, "dirty-hash: allocated oversized range escaped store clamp\n");
        return 1;
    }

    glBindBufferBase(GL_UNIFORM_BUFFER, binding_index, buffers[1]);
    glGetInteger64i_v(GL_UNIFORM_BUFFER_SIZE, binding_index, &indexed_size);
    slot = &ctx->active_state->buffer_base[_UNIFORM_BUFFER].buffers[binding_index];
    map.size = slot->size;
    if (glGetError() != GL_NO_ERROR || indexed_size != 0 ||
        mglBufferMapVisibleSize(&map) != 16) {
        fprintf(stderr, "dirty-hash: base binding did not use dynamic store size\n");
        return 1;
    }

    glBufferData(GL_UNIFORM_BUFFER, 8, NULL, GL_DYNAMIC_DRAW);
    if (glGetError() != GL_NO_ERROR || mglBufferMapVisibleSize(&map) != 8) {
        fprintf(stderr, "dirty-hash: base binding did not follow store resize\n");
        return 1;
    }

    glBindBufferBase(GL_UNIFORM_BUFFER, binding_index, 0);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    glDeleteBuffers(2, buffers);
    return 0;
}

static int verify_compute_short_range_copyback(void)
{
    static const char *compute_source =
        "#version 430 core\n"
        "layout(local_size_x=1) in;\n"
        "layout(std430, binding=0) buffer Data { uint values[2]; } data;\n"
        "void main() {\n"
        "  data.values[0] = 0x13572468u;\n"
        "  data.values[1] = 0xDEADBEEFu;\n"
        "}\n";
    const GLuint initial[2] = {0x01020304u, 0xA5A5A5A5u};
    const GLuint suffix_update = 0x2468ACE0u;
    GLuint result[2] = {0u, 0u};
    GLint saved_program = 0;
    GLuint shader = 0;
    GLuint program = 0;
    GLuint buffer = 0;
    int failed = 1;

    glGetIntegerv(GL_CURRENT_PROGRAM, &saved_program);
    while (glGetError() != GL_NO_ERROR) {}

    shader = compile_shader(GL_COMPUTE_SHADER, compute_source);
    if (!shader) {
        goto done;
    }
    program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDeleteShader(shader);
    shader = 0;

    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[2048] = {0};
        glGetProgramInfoLog(program, sizeof(log), NULL, log);
        fprintf(stderr, "dirty-hash: compute range program link failed: %s\n", log);
        goto done;
    }

    glGenBuffers(1, &buffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 sizeof(initial),
                 initial,
                 GL_DYNAMIC_COPY);
    glBindBufferRange(GL_SHADER_STORAGE_BUFFER,
                      0,
                      buffer,
                      0,
                      sizeof(GLuint));
    glUseProgram(program);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    glFinish();

    /* Do not map between the GPU write and this disjoint CPU update. The
     * latter must not upload a stale CPU snapshot over the copied-back
     * prefix. */
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                    sizeof(GLuint),
                    sizeof(suffix_update),
                    &suffix_update);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(result), result);

    GLenum error = glGetError();
    if (error != GL_NO_ERROR || result[0] != 0x13572468u ||
        result[1] != suffix_update) {
        fprintf(stderr,
                "dirty-hash: short SSBO range copyback lost GPU prefix or "
                "escaped range (error=0x%x first=0x%08x suffix=0x%08x)\n",
                (unsigned)error,
                result[0],
                result[1]);
        goto done;
    }

    failed = 0;

done:
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glUseProgram((GLuint)saved_program);
    if (buffer) glDeleteBuffers(1, &buffer);
    if (program) glDeleteProgram(program);
    if (shader) glDeleteShader(shader);
    return failed;
}

static int verify_compute_finish_visibility(void)
{
    static const char *compute_source =
        "#version 430 core\n"
        "layout(local_size_x=1) in;\n"
        "layout(std430, binding=0) buffer Data { uint values[4]; } data;\n"
        "void main() {\n"
        "  data.values[gl_WorkGroupID.x] = 0xC0DE0000u + gl_WorkGroupID.x;\n"
        "}\n";
    const GLuint initial[4] = {0u, 0u, 0u, 0u};
    GLuint observed[4] = {0u, 0u, 0u, 0u};
    GLint saved_program = 0;
    GLuint shader = 0;
    GLuint program = 0;
    GLuint buffer = 0;
    int failed = 1;

    glGetIntegerv(GL_CURRENT_PROGRAM, &saved_program);
    while (glGetError() != GL_NO_ERROR) {}

    shader = compile_shader(GL_COMPUTE_SHADER, compute_source);
    if (!shader) {
        goto done;
    }
    program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDeleteShader(shader);
    shader = 0;

    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[2048] = {0};
        glGetProgramInfoLog(program, sizeof(log), NULL, log);
        fprintf(stderr, "dirty-hash: compute finish program link failed: %s\n", log);
        goto done;
    }

    glGenBuffers(1, &buffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(initial), initial, GL_DYNAMIC_COPY);

    /* Full-buffer base binding: no short-range isolation, so the dispatch
       encodes no copy-back blit. The compute write is then the only work in
       the current command buffer when the finish-semantics flush runs, which
       is exactly what the empty-CB commit skip must not drop. */
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
    glUseProgram(program);
    glDispatchCompute(4, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    glFinish();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    const void *mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                                          0,
                                          sizeof(observed),
                                          GL_MAP_READ_BIT);
    if (mapped) memcpy(observed, mapped, sizeof(observed));
    if (mapped) glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    GLenum error = glGetError();
    size_t written_values = 0;
    for (size_t i = 0; i < 4; i++) {
        if (observed[i] == 0xC0DE0000u + (GLuint)i) {
            written_values++;
        }
    }
    if (!mapped || error != GL_NO_ERROR || written_values != 4u) {
        fprintf(stderr,
                "dirty-hash: compute dispatch lost across glFinish "
                "(error=0x%x written_values=%zu first=0x%08x)\n",
                (unsigned)error,
                written_values,
                observed[0]);
        goto done;
    }

    failed = 0;

done:
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glUseProgram((GLuint)saved_program);
    if (buffer) glDeleteBuffers(1, &buffer);
    if (program) glDeleteProgram(program);
    if (shader) glDeleteShader(shader);
    return failed;
}

static int verify_tcs_to_tes_short_range_visibility(void)
{
    static const char *vertex_source =
        "#version 430 core\n"
        "void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";
    static const char *control_source =
        "#version 430 core\n"
        "layout(vertices=1) out;\n"
        "layout(std430, binding=0) buffer Producer { uint value; } producer;\n"
        "void main() {\n"
        "  producer.value = 0x6A09E667u;\n"
        "  gl_TessLevelOuter[0] = 1.0;\n"
        "  gl_TessLevelOuter[1] = 1.0;\n"
        "  gl_TessLevelOuter[2] = 1.0;\n"
        "  gl_TessLevelInner[0] = 1.0;\n"
        "}\n";
    static const char *evaluation_source =
        "#version 430 core\n"
        "layout(triangles, equal_spacing, cw) in;\n"
        "layout(std140, binding=0) uniform Consumer { uvec2 values; } consumer;\n"
        "layout(std430, binding=1) buffer Result { uint values[2]; } result;\n"
        "void main() {\n"
        "  result.values[0] = consumer.values.x + uint(gl_PrimitiveID) * 0u;\n"
        "  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "}\n";
    static const char *fragment_source =
        "#version 430 core\n"
        "out vec4 color;\n"
        "void main() { color = vec4(1.0); }\n";
    const GLuint source_initial[2] = {0x01020304u, 0xA5A5A5A5u};
    const GLuint result_initial = 0u;
    GLuint source_result[2] = {0u, 0u};
    GLuint observed = 0u;
    GLuint shaders[4] = {0u, 0u, 0u, 0u};
    GLuint program = 0u;
    GLuint buffers[2] = {0u, 0u};
    GLuint vao = 0u;
    GLint saved_program = 0;
    GLint saved_vao = 0;
    GLint saved_patch_vertices = 3;
    GLboolean saved_rasterizer_discard = glIsEnabled(GL_RASTERIZER_DISCARD);
    int failed = 1;

    glGetIntegerv(GL_CURRENT_PROGRAM, &saved_program);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &saved_vao);
    glGetIntegerv(GL_PATCH_VERTICES, &saved_patch_vertices);
    while (glGetError() != GL_NO_ERROR) {}

    shaders[0] = compile_shader(GL_VERTEX_SHADER, vertex_source);
    shaders[1] = compile_shader(GL_TESS_CONTROL_SHADER, control_source);
    shaders[2] = compile_shader(GL_TESS_EVALUATION_SHADER, evaluation_source);
    shaders[3] = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    for (size_t i = 0; i < 4; i++) {
        if (!shaders[i]) {
            fprintf(stderr, "dirty-hash: TCS/TES visibility shader setup failed\n");
            goto done;
        }
    }

    program = glCreateProgram();
    for (size_t i = 0; i < 4; i++) glAttachShader(program, shaders[i]);
    glLinkProgram(program);
    for (size_t i = 0; i < 4; i++) {
        glDeleteShader(shaders[i]);
        shaders[i] = 0u;
    }

    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[2048] = {0};
        glGetProgramInfoLog(program, sizeof(log), NULL, log);
        fprintf(stderr, "dirty-hash: TCS/TES visibility program link failed: %s\n", log);
        goto done;
    }

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(2, buffers);

    /* TCS sees a four-byte std430 block and writes it directly. TES consumes
       the same four-byte range as a larger std140 block, forcing isolation.
       Its initialization must observe the pending TCS GPU write, not the
       source buffer's pre-dispatch CPU snapshot. */
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 sizeof(source_initial),
                 source_initial,
                 GL_DYNAMIC_COPY);
    glBindBufferRange(GL_SHADER_STORAGE_BUFFER,
                      0,
                      buffers[0],
                      0,
                      sizeof(GLuint));
    glBindBuffer(GL_UNIFORM_BUFFER, buffers[0]);
    glBindBufferRange(GL_UNIFORM_BUFFER,
                      0,
                      buffers[0],
                      0,
                      sizeof(GLuint));

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 sizeof(result_initial),
                 &result_initial,
                 GL_DYNAMIC_COPY);
    glBindBufferRange(GL_SHADER_STORAGE_BUFFER,
                      1,
                      buffers[1],
                      0,
                      sizeof(result_initial));

    glUseProgram(program);
    glPatchParameteri(GL_PATCH_VERTICES, 1);
    glEnable(GL_RASTERIZER_DISCARD);
    glDrawArrays(GL_PATCHES, 0, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    glFinish();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[1]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(observed), &observed);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[0]);
    const void *mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                                          0,
                                          sizeof(source_result),
                                          GL_MAP_READ_BIT);
    if (mapped) memcpy(source_result, mapped, sizeof(source_result));
    if (mapped) glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    GLenum error = glGetError();
    if (!mapped || error != GL_NO_ERROR || observed != 0x6A09E667u ||
        source_result[0] != 0x6A09E667u ||
        source_result[1] != source_initial[1]) {
        fprintf(stderr,
                "dirty-hash: TES isolated range saw stale TCS data "
                "(error=0x%x observed=0x%08x source=0x%08x suffix=0x%08x)\n",
                (unsigned)error,
                observed,
                source_result[0],
                source_result[1]);
        goto done;
    }

    failed = 0;

done:
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    glPatchParameteri(GL_PATCH_VERTICES, saved_patch_vertices);
    if (saved_rasterizer_discard) glEnable(GL_RASTERIZER_DISCARD);
    else glDisable(GL_RASTERIZER_DISCARD);
    glBindVertexArray((GLuint)saved_vao);
    glUseProgram((GLuint)saved_program);
    if (buffers[0] || buffers[1]) glDeleteBuffers(2, buffers);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    for (size_t i = 0; i < 4; i++) {
        if (shaders[i]) glDeleteShader(shaders[i]);
    }
    return failed;
}

static int verify_tess_factor_cpu_visibility(void)
{
    static const char *vertex_source =
        "#version 430 core\n"
        "void main() {\n"
        "  const vec2 p[3] = vec2[3](vec2(-0.5,-0.5), vec2(0.5,-0.5), vec2(0.0,0.5));\n"
        "  gl_Position = vec4(p[gl_VertexID % 3], 0.0, 1.0);\n"
        "}\n";
    static const char *control_source =
        "#version 430 core\n"
        "layout(vertices=3) out;\n"
        "void main() {\n"
        "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
        "  if (gl_InvocationID == 0) {\n"
        "    gl_TessLevelOuter[0] = 2.0;\n"
        "    gl_TessLevelOuter[1] = 2.0;\n"
        "    gl_TessLevelOuter[2] = 2.0;\n"
        "    gl_TessLevelInner[0] = 2.0;\n"
        "  }\n"
        "}\n";
    static const char *evaluation_source =
        "#version 430 core\n"
        "layout(triangles, equal_spacing, cw) in;\n"
        "layout(std430, binding=0) buffer Seen { uint values[12]; } seen;\n"
        "void main() {\n"
        "  uint index = uint(gl_TessCoord.x);\n"
        "  if (index < 12u) seen.values[index] = 0xA11CE000u + index;\n"
        "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position +\n"
        "                gl_TessCoord.y * gl_in[1].gl_Position +\n"
        "                gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "}\n";
    static const char *fragment_source =
        "#version 430 core\n"
        "out vec4 color;\n"
        "void main() { color = vec4(1.0); }\n";
    GLuint shaders[4] = {0u, 0u, 0u, 0u};
    GLuint program = 0u;
    GLuint vao = 0u;
    GLuint seen_buffer = 0u;
    GLuint generated_query = 0u;
    GLuint observed[12] = {0u};
    const GLuint initial[12] = {0u};
    GLint saved_program = 0;
    GLint saved_vao = 0;
    GLint saved_patch_vertices = 3;
    GLboolean saved_rasterizer_discard = glIsEnabled(GL_RASTERIZER_DISCARD);
    int failed = 1;

    glGetIntegerv(GL_CURRENT_PROGRAM, &saved_program);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &saved_vao);
    glGetIntegerv(GL_PATCH_VERTICES, &saved_patch_vertices);
    while (glGetError() != GL_NO_ERROR) {}

    shaders[0] = compile_shader(GL_VERTEX_SHADER, vertex_source);
    shaders[1] = compile_shader(GL_TESS_CONTROL_SHADER, control_source);
    shaders[2] = compile_shader(GL_TESS_EVALUATION_SHADER, evaluation_source);
    shaders[3] = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    for (size_t i = 0; i < 4; i++) {
        if (!shaders[i]) {
            fprintf(stderr, "dirty-hash: tess-factor visibility shader setup failed\n");
            goto done;
        }
    }

    program = glCreateProgram();
    for (size_t i = 0; i < 4; i++) glAttachShader(program, shaders[i]);
    glLinkProgram(program);
    for (size_t i = 0; i < 4; i++) {
        glDeleteShader(shaders[i]);
        shaders[i] = 0u;
    }

    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[2048] = {0};
        glGetProgramInfoLog(program, sizeof(log), NULL, log);
        fprintf(stderr, "dirty-hash: tess-factor visibility program link failed: %s\n", log);
        goto done;
    }

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &seen_buffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, seen_buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(initial), initial, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, seen_buffer);
    glGenQueries(1, &generated_query);

    glUseProgram(program);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginQuery(GL_PRIMITIVES_GENERATED, generated_query);
    glDrawArrays(GL_PATCHES, 0, 3);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    glFinish();

    GLuint64 primitives_generated = ~(GLuint64)0;
    glGetQueryObjectui64v(generated_query,
                          GL_QUERY_RESULT,
                          &primitives_generated);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, seen_buffer);
    const void *mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                                          0,
                                          sizeof(observed),
                                          GL_MAP_READ_BIT);
    if (mapped) memcpy(observed, mapped, sizeof(observed));
    if (mapped) glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    GLenum error = glGetError();

    size_t written_values = 0;
    for (size_t i = 0; i < 12; i++) {
        if (observed[i] == 0xA11CE000u + (GLuint)i) {
            written_values++;
        }
    }
    if (!mapped || error != GL_NO_ERROR ||
        primitives_generated != 4u || written_values != 12u) {
        fprintf(stderr,
                "dirty-hash: TES saw stale TCS tess factors "
                "(error=0x%x generated=%llu written_values=%zu)\n",
                (unsigned)error,
                (unsigned long long)primitives_generated,
                written_values);
        goto done;
    }

    failed = 0;

done:
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glPatchParameteri(GL_PATCH_VERTICES, saved_patch_vertices);
    if (saved_rasterizer_discard) glEnable(GL_RASTERIZER_DISCARD);
    else glDisable(GL_RASTERIZER_DISCARD);
    glBindVertexArray((GLuint)saved_vao);
    glUseProgram((GLuint)saved_program);
    if (generated_query) glDeleteQueries(1, &generated_query);
    if (seen_buffer) glDeleteBuffers(1, &seen_buffer);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    for (size_t i = 0; i < 4; i++) {
        if (shaders[i]) glDeleteShader(shaders[i]);
    }
    return failed;
}

static int verify_tes_xfb_range_isolation(void)
{
    GLint saved_program = 0;
    GLint saved_vao = 0;
    GLboolean saved_rasterizer_discard = glIsEnabled(GL_RASTERIZER_DISCARD);
    glGetIntegerv(GL_CURRENT_PROGRAM, &saved_program);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &saved_vao);

    static const char *vertex_source =
        "#version 410 core\n"
        "void main() {\n"
        "  const vec2 p[3] = vec2[3](vec2(-0.5,-0.5), vec2(0.5,-0.5), vec2(0.0,0.5));\n"
        "  gl_Position = vec4(p[gl_VertexID % 3], 0.0, 1.0);\n"
        "}\n";
    static const char *control_source =
        "#version 410 core\n"
        "layout(vertices=3) out;\n"
        "void main() {\n"
        "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
        "  if (gl_InvocationID == 0) {\n"
        "    gl_TessLevelOuter[0] = 1.0;\n"
        "    gl_TessLevelOuter[1] = 1.0;\n"
        "    gl_TessLevelOuter[2] = 1.0;\n"
        "    gl_TessLevelInner[0] = 1.0;\n"
        "  }\n"
        "}\n";
    static const char *evaluation_source =
        "#version 410 core\n"
        "layout(triangles, equal_spacing, cw) in;\n"
        "layout(location=0) out vec4 captured;\n"
        "void main() {\n"
        "  captured = vec4(gl_TessCoord.x, gl_TessCoord.x + 10.0,\n"
        "                  gl_TessCoord.x + 20.0, gl_TessCoord.x + 30.0);\n"
        "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position +\n"
        "                gl_TessCoord.y * gl_in[1].gl_Position +\n"
        "                gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "}\n";
    static const char *fragment_source =
        "#version 410 core\n"
        "out vec4 color;\n"
        "void main() { color = vec4(1.0); }\n";
    GLuint shaders[4] = {
        compile_shader(GL_VERTEX_SHADER, vertex_source),
        compile_shader(GL_TESS_CONTROL_SHADER, control_source),
        compile_shader(GL_TESS_EVALUATION_SHADER, evaluation_source),
        compile_shader(GL_FRAGMENT_SHADER, fragment_source),
    };
    for (size_t i = 0; i < 4; i++) {
        if (!shaders[i]) {
            fprintf(stderr, "dirty-hash: TES XFB shader setup failed\n");
            return 1;
        }
    }

    GLuint program = glCreateProgram();
    for (size_t i = 0; i < 4; i++) glAttachShader(program, shaders[i]);
    const char *varying = "captured";
    glTransformFeedbackVaryings(program, 1, &varying, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(program);
    for (size_t i = 0; i < 4; i++) glDeleteShader(shaders[i]);

    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[2048] = {0};
        glGetProgramInfoLog(program, sizeof(log), NULL, log);
        fprintf(stderr, "dirty-hash: TES XFB program link failed: %s\n", log);
        glDeleteProgram(program);
        return 1;
    }

    unsigned char canary[224];
    unsigned char result[224];
    static const GLfloat level_one_capture[12] = {
        0.0f, 10.0f, 20.0f, 30.0f,
        1.0f, 11.0f, 21.0f, 31.0f,
        2.0f, 12.0f, 22.0f, 32.0f,
    };
    memset(canary, 0xA5, sizeof(canary));
    memset(result, 0, sizeof(result));

    GLuint vao = 0;
    GLuint xfb = 0;
    GLuint written_query = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &xfb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof(canary), canary, GL_STATIC_READ);

    /* Two triangle primitives need 2 * 3 * sizeof(vec4) = 96 bytes. The
       48-byte range holds exactly the first primitive; bytes after the range
       must retain their canary values. */
    glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xfb, 16, 48);
    glUseProgram(program);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glEnable(GL_RASTERIZER_DISCARD);
    glGenQueries(1, &written_query);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, written_query);
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_PATCHES, 0, 6);
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glDisable(GL_RASTERIZER_DISCARD);
    glFinish();

    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb);
    const void *mapped = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER,
                                          0,
                                          sizeof(result),
                                          GL_MAP_READ_BIT);
    if (mapped) memcpy(result, mapped, sizeof(result));
    if (mapped) glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);
    GLuint64 primitives_written = ~(GLuint64)0;
    glGetQueryObjectui64v(written_query, GL_QUERY_RESULT, &primitives_written);
    GLenum error = glGetError();
    int prefix_changed = memcmp(canary, result, 16) != 0;
    int suffix_changed = memcmp(canary + 64, result + 64, sizeof(canary) - 64) != 0;
    int range_mismatch = memcmp(level_one_capture, result + 16,
                                sizeof(level_one_capture)) != 0;
    int failed = !mapped || error != GL_NO_ERROR || primitives_written != 1 ||
                 prefix_changed || suffix_changed || range_mismatch;
    if (failed) {
        fprintf(stderr,
                "dirty-hash: TES XFB escaped its range "
                "(error=0x%x written=%llu prefix=%d suffix=%d range=%d)\n",
                (unsigned)error,
                (unsigned long long)primitives_written,
                prefix_changed,
                suffix_changed,
                range_mismatch);
    }

    if (!failed) {
        /* The first draw fits directly. The second draw requests two
           primitives, but only one complete primitive remains, so it must use
           the bounded temporary path and append at the session cursor. */
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb);
        glBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(canary), canary);
        glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xfb, 16, 96);
        memset(result, 0, sizeof(result));

        glEnable(GL_RASTERIZER_DISCARD);
        glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, written_query);
        glBeginTransformFeedback(GL_TRIANGLES);
        glDrawArrays(GL_PATCHES, 0, 3);
        glPauseTransformFeedback();
        glResumeTransformFeedback();
        glDrawArrays(GL_PATCHES, 0, 6);
        glEndTransformFeedback();
        glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
        glDisable(GL_RASTERIZER_DISCARD);
        glFinish();

        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb);
        mapped = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER,
                                  0,
                                  sizeof(result),
                                  GL_MAP_READ_BIT);
        if (mapped) memcpy(result, mapped, sizeof(result));
        if (mapped) glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);
        primitives_written = ~(GLuint64)0;
        glGetQueryObjectui64v(written_query,
                              GL_QUERY_RESULT,
                              &primitives_written);
        error = glGetError();

        prefix_changed = memcmp(canary, result, 16) != 0;
        suffix_changed = memcmp(canary + 112,
                                result + 112,
                                sizeof(canary) - 112) != 0;
        int first_capture_mismatch =
            memcmp(level_one_capture, result + 16,
                   sizeof(level_one_capture)) != 0;
        int second_capture_mismatch =
            memcmp(level_one_capture, result + 64,
                   sizeof(level_one_capture)) != 0;
        failed = !mapped || error != GL_NO_ERROR ||
                 primitives_written != 2 || prefix_changed || suffix_changed ||
                 first_capture_mismatch || second_capture_mismatch;
        if (failed) {
            fprintf(stderr,
                    "dirty-hash: TES XFB session did not append across draws "
                    "(error=0x%x written=%llu prefix=%d suffix=%d first=%d second=%d)\n",
                    (unsigned)error,
                    (unsigned long long)primitives_written,
                    prefix_changed,
                    suffix_changed,
                    first_capture_mismatch,
                    second_capture_mismatch);
        }
    }

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
    glBindVertexArray((GLuint)saved_vao);
    glUseProgram((GLuint)saved_program);
    if (saved_rasterizer_discard) glEnable(GL_RASTERIZER_DISCARD);
    glDeleteBuffers(1, &xfb);
    glDeleteQueries(1, &written_query);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
    return failed;
}

static int floats_close(GLfloat a, GLfloat b)
{
    GLfloat d = a - b;
    if (d < 0.0f) d = -d;
    return d < 1e-5f;
}

/* GetUniform must read per-location slots (arr[i] at location+i) and unpack
 * Metal-packed mat3 (12 words) back to GL's 9-float column-major layout.
 * Also covers mat3 array elements written via UniformMatrix3fv(loc+1), which
 * store tightly packed 9 floats when the packing helper only matches base. */
static int verify_get_uniform_array_and_mat3(void)
{
    static const char *vertex_source =
        "#version 330 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *fragment_source =
        "#version 330 core\n"
        "uniform float u_arr[4];\n"
        "uniform mat3 u_m;\n"
        "uniform mat3 u_marr[2];\n"
        "out vec4 color;\n"
        "void main() {\n"
        "  color = vec4(u_arr[0] + u_arr[1] + u_arr[2] + u_arr[3],\n"
        "               u_m[0][0] + u_m[1][1] + u_m[2][2],\n"
        "               u_marr[0][0][0] + u_marr[1][1][1], 1.0);\n"
        "}\n";
    static const GLfloat arr_per_elem[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    static const GLfloat arr_bulk[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    static const GLfloat mat_a[9] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
    };
    static const GLfloat mat_b[9] = {
        0.5f, 1.5f, 2.5f,
        3.5f, 4.5f, 5.5f,
        6.5f, 7.5f, 8.5f,
    };

    GLint saved_program = 0;
    GLuint vs = 0, fs = 0, program = 0;
    int failed = 1;
    GLfloat got[9];
    GLint loc_arr = -1, loc_m = -1, loc_marr = -1;
    int i;

    glGetIntegerv(GL_CURRENT_PROGRAM, &saved_program);
    while (glGetError() != GL_NO_ERROR) {}

    vs = compile_shader(GL_VERTEX_SHADER, vertex_source);
    fs = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    if (!vs || !fs) {
        goto done;
    }
    program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);
    vs = fs = 0;

    {
        GLint linked = GL_FALSE;
        glGetProgramiv(program, GL_LINK_STATUS, &linked);
        if (!linked) {
            char log[2048] = {0};
            glGetProgramInfoLog(program, sizeof(log), NULL, log);
            fprintf(stderr, "dirty-hash: GetUniform program link failed: %s\n", log);
            goto done;
        }
    }

    glUseProgram(program);
    loc_arr = glGetUniformLocation(program, "u_arr[0]");
    if (loc_arr < 0) {
        loc_arr = glGetUniformLocation(program, "u_arr");
    }
    loc_m = glGetUniformLocation(program, "u_m");
    loc_marr = glGetUniformLocation(program, "u_marr[0]");
    if (loc_marr < 0) {
        loc_marr = glGetUniformLocation(program, "u_marr");
    }
    if (loc_arr < 0 || loc_m < 0 || loc_marr < 0) {
        fprintf(stderr,
                "dirty-hash: GetUniform missing locations arr=%d m=%d marr=%d\n",
                loc_arr, loc_m, loc_marr);
        goto done;
    }

    /* Bulk count>1 upload into the base slot; GetUniform(loc+i) must stride. */
    glUniform1fv(loc_arr, 4, arr_bulk);
    for (i = 0; i < 4; i++) {
        GLfloat v = -1.0f;
        glGetUniformfv(program, loc_arr + i, &v);
        if (!floats_close(v, arr_bulk[i])) {
            fprintf(stderr,
                    "dirty-hash: GetUniform float arr[%d] bulk got %g want %g\n",
                    i, (double)v, (double)arr_bulk[i]);
            goto done;
        }
    }

    /* Per-element writes: each location is its own CPU slot. */
    for (i = 0; i < 4; i++) {
        glUniform1f(loc_arr + i, arr_per_elem[i]);
    }
    for (i = 0; i < 4; i++) {
        GLfloat v = -1.0f;
        glGetUniformfv(program, loc_arr + i, &v);
        if (!floats_close(v, arr_per_elem[i])) {
            fprintf(stderr,
                    "dirty-hash: GetUniform float arr[%d] per-elem got %g want %g\n",
                    i, (double)v, (double)arr_per_elem[i]);
            goto done;
        }
    }

    /* Scalar mat3 at base: Metal-packed 12-word store must round-trip as 9. */
    glUniformMatrix3fv(loc_m, 1, GL_FALSE, mat_a);
    memset(got, 0, sizeof(got));
    glGetUniformfv(program, loc_m, got);
    for (i = 0; i < 9; i++) {
        if (!floats_close(got[i], mat_a[i])) {
            fprintf(stderr,
                    "dirty-hash: GetUniform mat3[%d] got %g want %g\n",
                    i, (double)got[i], (double)mat_a[i]);
            goto done;
        }
    }

    /* mat3[2] bulk at base (packed 24 words); element 1 is a stride read. */
    {
        GLfloat both[18];
        memcpy(both, mat_a, 9u * sizeof(GLfloat));
        memcpy(both + 9, mat_b, 9u * sizeof(GLfloat));
        glUniformMatrix3fv(loc_marr, 2, GL_FALSE, both);
        memset(got, 0, sizeof(got));
        glGetUniformfv(program, loc_marr + 1, got);
        for (i = 0; i < 9; i++) {
            if (!floats_close(got[i], mat_b[i])) {
                fprintf(stderr,
                        "dirty-hash: GetUniform mat3 arr bulk[1][%d] got %g want %g\n",
                        i, (double)got[i], (double)mat_b[i]);
                goto done;
            }
        }
    }

    /* mat3 array element 1 written alone (often tightly packed 9 floats). */
    glUniformMatrix3fv(loc_marr + 1, 1, GL_FALSE, mat_b);
    memset(got, 0, sizeof(got));
    glGetUniformfv(program, loc_marr + 1, got);
    for (i = 0; i < 9; i++) {
        if (!floats_close(got[i], mat_b[i])) {
            fprintf(stderr,
                    "dirty-hash: GetUniform mat3 arr[1][%d] got %g want %g\n",
                    i, (double)got[i], (double)mat_b[i]);
            goto done;
        }
    }

    {
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            fprintf(stderr, "dirty-hash: GetUniform left GL error 0x%x\n",
                    (unsigned)error);
            goto done;
        }
    }

    failed = 0;

done:
    glUseProgram((GLuint)saved_program);
    if (program) glDeleteProgram(program);
    if (vs) glDeleteShader(vs);
    if (fs) glDeleteShader(fs);
    return failed;
}

static int verify_air_native_tessellation_draw(void)
{
    static const char *vertex_source =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position.x - 2.0, position.y, 0.0, 1.0); }\n";
    static const char *control_source =
        "#version 450 core\n"
        "layout(vertices=3) out;\n"
        "void main() {\n"
        "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
        "  gl_TessLevelOuter[0] = 1.0;\n"
        "  gl_TessLevelOuter[1] = 1.0;\n"
        "  gl_TessLevelOuter[2] = 1.0;\n"
        "  gl_TessLevelOuter[3] = 1.0;\n"
        "  gl_TessLevelInner[0] = 1.0;\n"
        "  gl_TessLevelInner[1] = 1.0;\n"
        "}\n";
    static const char *evaluation_source =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, cw) in;\n"
        "layout(location=0) out vec4 tessColor;\n"
        "void main() {\n"
        "  gl_Position = gl_in[0].gl_Position * gl_TessCoord.x +\n"
        "                gl_in[1].gl_Position * gl_TessCoord.y +\n"
        "                gl_in[2].gl_Position * gl_TessCoord.z;\n"
        "  tessColor = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "}\n";
    static const char *fragment_source =
        "#version 450 core\n"
        "layout(location=0) in vec4 tessColor;\n"
        "out vec4 color;\n"
        "void main() { color = tessColor; }\n";
    static const GLfloat vertices[6] = {
        1.2f, -0.8f, 2.8f, -0.8f, 2.0f, 0.8f,
    };
    GLuint shaders[4] = {0u, 0u, 0u, 0u};
    GLuint program = 0u;
    GLuint program_no_tcs = 0u;
    GLuint vao = 0u;
    GLuint buffer = 0u;
    GLint saved_program = 0;
    GLint saved_vao = 0;
    GLint saved_patch_vertices = 3;
    GLboolean saved_cull = glIsEnabled(GL_CULL_FACE);
    GLboolean saved_discard = glIsEnabled(GL_RASTERIZER_DISCARD);
    int failed = 1;

    glGetIntegerv(GL_CURRENT_PROGRAM, &saved_program);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &saved_vao);
    glGetIntegerv(GL_PATCH_VERTICES, &saved_patch_vertices);
    shaders[0] = compile_shader(GL_VERTEX_SHADER, vertex_source);
    shaders[1] = compile_shader(GL_TESS_CONTROL_SHADER, control_source);
    shaders[2] = compile_shader(GL_TESS_EVALUATION_SHADER, evaluation_source);
    shaders[3] = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    for (size_t i = 0; i < 4; i++) {
        if (!shaders[i]) goto done;
    }

    program = glCreateProgram();
    for (size_t i = 0; i < 4; i++) glAttachShader(program, shaders[i]);
    glLinkProgram(program);
    for (size_t i = 0; i < 4; i++) {
        glDeleteShader(shaders[i]);
        shaders[i] = 0u;
    }
    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[2048] = {0};
        glGetProgramInfoLog(program, sizeof(log), NULL, log);
        fprintf(stderr, "dirty-hash: AIR native tess program link failed: %s\n", log);
        goto done;
    }

    vao = make_vao(vertices, &buffer);
    glUseProgram(program);
    glBindVertexArray(vao);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glDisable(GL_CULL_FACE);
    glDisable(GL_RASTERIZER_DISCARD);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();

    GLubyte center[4] = {0u, 0u, 0u, 0u};
    glReadPixels(TEST_W / 2, TEST_H / 2, 1, 1,
                 GL_RGBA, GL_UNSIGNED_BYTE, center);
    if (glGetError() != GL_NO_ERROR || center[0] > 32u ||
        center[1] < 200u || center[2] > 32u) {
        fprintf(stderr,
                "dirty-hash: AIR native tess draw missing center=%u,%u,%u,%u\n",
                center[0], center[1], center[2], center[3]);
        goto done;
    }

    shaders[0] = compile_shader(GL_VERTEX_SHADER, vertex_source);
    shaders[1] = compile_shader(GL_TESS_EVALUATION_SHADER, evaluation_source);
    shaders[2] = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    if (!shaders[0] || !shaders[1] || !shaders[2]) goto done;
    program_no_tcs = glCreateProgram();
    for (size_t i = 0; i < 3; i++) glAttachShader(program_no_tcs, shaders[i]);
    glLinkProgram(program_no_tcs);
    for (size_t i = 0; i < 3; i++) {
        glDeleteShader(shaders[i]);
        shaders[i] = 0u;
    }
    linked = GL_FALSE;
    glGetProgramiv(program_no_tcs, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[2048] = {0};
        glGetProgramInfoLog(program_no_tcs, sizeof(log), NULL, log);
        fprintf(stderr, "dirty-hash: AIR TES-only program link failed: %s\n", log);
        goto done;
    }
    glUseProgram(program_no_tcs);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();
    memset(center, 0, sizeof(center));
    glReadPixels(TEST_W / 2, TEST_H / 2, 1, 1,
                 GL_RGBA, GL_UNSIGNED_BYTE, center);
    if (glGetError() != GL_NO_ERROR || center[0] > 32u ||
        center[1] < 200u || center[2] > 32u) {
        fprintf(stderr,
                "dirty-hash: AIR TES-only draw missing center=%u,%u,%u,%u\n",
                center[0], center[1], center[2], center[3]);
        goto done;
    }
    printf("AIR_NATIVE_TESS_OK tcs=1 no_tcs=1 center=%u,%u,%u,%u\n",
           center[0], center[1], center[2], center[3]);
    failed = 0;

done:
    glPatchParameteri(GL_PATCH_VERTICES, saved_patch_vertices);
    if (saved_cull) glEnable(GL_CULL_FACE); else glDisable(GL_CULL_FACE);
    if (saved_discard) glEnable(GL_RASTERIZER_DISCARD);
    else glDisable(GL_RASTERIZER_DISCARD);
    glBindVertexArray((GLuint)saved_vao);
    glUseProgram((GLuint)saved_program);
    if (buffer) glDeleteBuffers(1, &buffer);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    if (program_no_tcs) glDeleteProgram(program_no_tcs);
    for (size_t i = 0; i < 4; i++) {
        if (shaders[i]) glDeleteShader(shaders[i]);
    }
    return failed;
}

static int verify_air_geometry_compute_draw(void)
{
    static const char *vertex_source =
        "#version 450 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position.x - 2.0, position.y, 0.0, 1.0); }\n";
    static const char *geometry_source =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices=6) out;\n"
        "void main() {\n"
        "  gl_Position = gl_in[0].gl_Position + vec4(2.0, 0.0, 0.0, 0.0); EmitVertex();\n"
        "  gl_Position = gl_in[1].gl_Position + vec4(2.0, 0.0, 0.0, 0.0); EmitVertex();\n"
        "  gl_Position = gl_in[2].gl_Position + vec4(2.0, 0.0, 0.0, 0.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "  gl_Position = gl_in[2].gl_Position + vec4(2.0, 0.0, 0.0, 0.0); EmitVertex();\n"
        "  gl_Position = gl_in[1].gl_Position + vec4(2.0, 0.0, 0.0, 0.0); EmitVertex();\n"
        "  gl_Position = gl_in[0].gl_Position + vec4(2.0, 0.0, 0.0, 0.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    static const char *fragment_source =
        "#version 450 core\n"
        "out vec4 color;\n"
        "void main() { color = vec4(0.0, 0.0, 1.0, 1.0); }\n";
    static const GLfloat vertices[6] = {
        -0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f,
    };
    GLuint shaders[3] = {0u, 0u, 0u};
    GLuint program = 0u;
    GLuint vao = 0u;
    GLuint buffer = 0u;
    GLint saved_program = 0;
    GLint saved_vao = 0;
    GLboolean saved_cull = glIsEnabled(GL_CULL_FACE);
    GLboolean saved_discard = glIsEnabled(GL_RASTERIZER_DISCARD);
    int failed = 1;

    glGetIntegerv(GL_CURRENT_PROGRAM, &saved_program);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &saved_vao);
    shaders[0] = compile_shader(GL_VERTEX_SHADER, vertex_source);
    shaders[1] = compile_shader(GL_GEOMETRY_SHADER, geometry_source);
    shaders[2] = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    for (size_t i = 0; i < 3; i++) {
        if (!shaders[i]) goto done;
    }

    program = glCreateProgram();
    for (size_t i = 0; i < 3; i++) glAttachShader(program, shaders[i]);
    glLinkProgram(program);
    for (size_t i = 0; i < 3; i++) {
        glDeleteShader(shaders[i]);
        shaders[i] = 0u;
    }
    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[2048] = {0};
        glGetProgramInfoLog(program, sizeof(log), NULL, log);
        fprintf(stderr, "dirty-hash: AIR GS program link failed: %s\n", log);
        goto done;
    }

    vao = make_vao(vertices, &buffer);
    glUseProgram(program);
    glBindVertexArray(vao);
    glDisable(GL_CULL_FACE);
    glDisable(GL_RASTERIZER_DISCARD);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    /* Repeat the same program/stage/function tuple so the Metal-cpp A/B gate
     * proves the renderer-owned compute PSO cache is reused. */
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();

    GLubyte center[4] = {0u, 0u, 0u, 0u};
    glReadPixels(TEST_W / 2, TEST_H / 2, 1, 1,
                 GL_RGBA, GL_UNSIGNED_BYTE, center);
    if (glGetError() != GL_NO_ERROR || center[0] > 32u ||
        center[1] > 32u || center[2] < 200u) {
        fprintf(stderr,
                "dirty-hash: AIR GS compute draw missing center=%u,%u,%u,%u\n",
                center[0], center[1], center[2], center[3]);
        goto done;
    }
    printf("AIR_GS_COMPUTE_OK center=%u,%u,%u,%u\n",
           center[0], center[1], center[2], center[3]);
    failed = 0;

done:
    if (saved_cull) glEnable(GL_CULL_FACE); else glDisable(GL_CULL_FACE);
    if (saved_discard) glEnable(GL_RASTERIZER_DISCARD);
    else glDisable(GL_RASTERIZER_DISCARD);
    glBindVertexArray((GLuint)saved_vao);
    glUseProgram((GLuint)saved_program);
    if (buffer) glDeleteBuffers(1, &buffer);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    for (size_t i = 0; i < 3; i++) {
        if (shaders[i]) glDeleteShader(shaders[i]);
    }
    return failed;
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
    if (getenv("MGL_USE_AIR")) {
        if (verify_air_aux_render_pipelines() != 0) {
            return 1;
        }
        if (getenv("MGL_TEST_AIR_AUX_RENDER_ONLY")) {
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
            return 0;
        }
        if (verify_air_geometry_compute_draw() != 0) {
            return 1;
        }
        if (getenv("MGL_TEST_AIR_GS_ONLY")) {
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
            return 0;
        }
        if (verify_air_native_tessellation_draw() != 0) {
            return 1;
        }
        if (getenv("MGL_TEST_AIR_NATIVE_TESS_ONLY")) {
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
            return 0;
        }
    }
    glEnable(GL_CULL_FACE);
    if (verify_stable_hash_cache(ctx) != 0) {
        return 1;
    }
    if (verify_transform_feedback_binding_hash(ctx) != 0) {
        return 1;
    }
    if (verify_buffer_range_lifecycle(ctx) != 0) {
        return 1;
    }
    if (verify_compute_short_range_copyback() != 0) {
        return 1;
    }
    if (verify_compute_finish_visibility() != 0) {
        return 1;
    }
    if (!getenv("MGL_USE_AIR")) {
        if (verify_tcs_to_tes_short_range_visibility() != 0) {
            return 1;
        }
        if (verify_tess_factor_cpu_visibility() != 0) {
            return 1;
        }
        if (verify_tes_xfb_range_isolation() != 0) {
            return 1;
        }
    }
    if (verify_get_uniform_array_and_mat3() != 0) {
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
