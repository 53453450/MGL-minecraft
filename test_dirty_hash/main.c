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
    if (verify_buffer_range_lifecycle(ctx) != 0) {
        return 1;
    }
    if (verify_compute_short_range_copyback() != 0) {
        return 1;
    }
    if (verify_tcs_to_tes_short_range_visibility() != 0) {
        return 1;
    }
    if (verify_tess_factor_cpu_visibility() != 0) {
        return 1;
    }
    if (verify_tes_xfb_range_isolation() != 0) {
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
