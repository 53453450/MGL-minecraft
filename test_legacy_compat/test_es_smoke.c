/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * OpenGL ES 3.2 smoke + minimum CTS subset: context, GLES 3.2 limits table,
 * desktop-only Get rejection, GLSL ES 3.20 compile/link, DrawArrays.
 */

#include <stdio.h>
#include <string.h>
#include <GL/glcorearb.h>
#include "glm_context.h"

static int fail_count;

static void expect(int cond, const char *msg)
{
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        fail_count++;
    } else {
        fprintf(stderr, "PASS: %s\n", msg);
    }
}

static void expect_ge(GLint value, GLint min_value, const char *msg)
{
    if (value < min_value) {
        fprintf(stderr, "FAIL: %s (got %d, min %d)\n", msg, (int)value, (int)min_value);
        fail_count++;
    } else {
        fprintf(stderr, "PASS: %s (%d)\n", msg, (int)value);
    }
}

static GLuint compile_stage(GLenum type, const char *src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        memset(log, 0, sizeof(log));
        glGetShaderInfoLog(shader, sizeof(log), NULL, log);
        fprintf(stderr, "FAIL: ES shader compile: %s\n", log);
        fail_count++;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

int main(void)
{
    fail_count = 0;
    GLMContext ctx = createGLMContext(GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV,
                                      GL_DEPTH_COMPONENT, GL_FLOAT, 0, 0);
    expect(ctx != NULL, "ES createGLMContext");
    if (!ctx)
        return 1;
    MGLsetCurrentContext(ctx);
    while (glGetError() != GL_NO_ERROR) {
    }

    const GLubyte *ver = glGetString(GL_VERSION);
    expect(ver && strstr((const char *)ver, "ES 3.2") != NULL,
           "ES GL_VERSION advertises 3.2");
    const GLubyte *sl = glGetString(GL_SHADING_LANGUAGE_VERSION);
    expect(sl && strstr((const char *)sl, "3.20") != NULL,
           "ES SHADING_LANGUAGE_VERSION advertises 3.20");

    GLint major = 0, minor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    expect(major == 3 && minor == 2, "ES MAJOR/MINOR 3.2");

    while (glGetError() != GL_NO_ERROR) {
    }
    GLint profile = 0;
    glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &profile);
    expect(glGetError() == GL_INVALID_ENUM,
           "ES CONTEXT_PROFILE_MASK is INVALID_ENUM");

    GLint value = 0;
    glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &value);
    expect_ge(value, 16, "ES MAX_VERTEX_ATTRIBS");
    glGetIntegerv(GL_MAX_VERTEX_UNIFORM_VECTORS, &value);
    expect_ge(value, 256, "ES MAX_VERTEX_UNIFORM_VECTORS");
    glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_VECTORS, &value);
    expect_ge(value, 224, "ES MAX_FRAGMENT_UNIFORM_VECTORS");
    glGetIntegerv(GL_MAX_VARYING_VECTORS, &value);
    expect_ge(value, 15, "ES MAX_VARYING_VECTORS");
    glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &value);
    expect_ge(value, 16, "ES MAX_TEXTURE_IMAGE_UNITS");
    glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &value);
    expect_ge(value, 48, "ES MAX_COMBINED_TEXTURE_IMAGE_UNITS");
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &value);
    expect_ge(value, 2048, "ES MAX_TEXTURE_SIZE");
    glGetIntegerv(GL_MAX_CUBE_MAP_TEXTURE_SIZE, &value);
    expect_ge(value, 2048, "ES MAX_CUBE_MAP_TEXTURE_SIZE");
    glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &value);
    expect_ge(value, 256, "ES MAX_3D_TEXTURE_SIZE");
    glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &value);
    expect_ge(value, 256, "ES MAX_ARRAY_TEXTURE_LAYERS");
    glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &value);
    expect_ge(value, 2048, "ES MAX_RENDERBUFFER_SIZE");
    glGetIntegerv(GL_MAX_DRAW_BUFFERS, &value);
    expect_ge(value, 4, "ES MAX_DRAW_BUFFERS");
    glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &value);
    expect_ge(value, 4, "ES MAX_COLOR_ATTACHMENTS");
    glGetIntegerv(GL_MAX_SAMPLES, &value);
    expect(value == 4, "ES MAX_SAMPLES matches Metal overlay (4)");
    glGetIntegerv(GL_MAX_IMAGE_SAMPLES, &value);
    expect(value == 4, "ES MAX_IMAGE_SAMPLES == MAX_SAMPLES");
    glGetIntegerv(GL_MAX_UNIFORM_BUFFER_BINDINGS, &value);
    expect_ge(value, 24, "ES MAX_UNIFORM_BUFFER_BINDINGS");
    glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &value);
    expect_ge(value, 16384, "ES MAX_UNIFORM_BLOCK_SIZE");
    glGetIntegerv(GL_MAX_TESS_GEN_LEVEL, &value);
    expect_ge(value, 64, "ES MAX_TESS_GEN_LEVEL");
    glGetIntegerv(GL_MAX_PATCH_VERTICES, &value);
    expect_ge(value, 32, "ES MAX_PATCH_VERTICES");
    glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES, &value);
    expect_ge(value, 256, "ES MAX_GEOMETRY_OUTPUT_VERTICES");

    static const char *kVS =
        "#version 320 es\n"
        "layout(location = 0) in vec4 a_position;\n"
        "void main() { gl_Position = a_position; }\n";
    static const char *kFS =
        "#version 320 es\n"
        "precision mediump float;\n"
        "layout(location = 0) out vec4 o_color;\n"
        "void main() { o_color = vec4(1.0); }\n";
    GLuint vs = compile_stage(GL_VERTEX_SHADER, kVS);
    GLuint fs = compile_stage(GL_FRAGMENT_SHADER, kFS);
    if (vs && fs) {
        GLuint program = glCreateProgram();
        glAttachShader(program, vs);
        glAttachShader(program, fs);
        glLinkProgram(program);
        GLint linked = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &linked);
        expect(linked, "ES GLSL 3.20 program links");
        if (linked) {
            glUseProgram(program);
            expect(glGetError() == GL_NO_ERROR, "ES UseProgram");
        }
        glDeleteProgram(program);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    glDrawArrays(GL_TRIANGLES, 0, 3);
    GLenum err = glGetError();
    expect(err == GL_NO_ERROR || err == GL_INVALID_OPERATION,
           "ES DrawArrays does not crash");

    destroyGLMContext(ctx);
    if (fail_count) {
        fprintf(stderr, "es-smoke: %d failure(s)\n", fail_count);
        return 1;
    }
    fprintf(stderr, "es-smoke: ok\n");
    return 0;
}
