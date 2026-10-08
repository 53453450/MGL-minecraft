/*
 * Independent probe: one compute dispatch that load+stores 16 distinct
 * storage images (MAX_IMAGE_UNITS advertisement self-check).
 */

#include <stdio.h>
#include <string.h>
#include <GL/glcorearb.h>
#include "glm_context.h"
#include "mgl_platform_shell_result.h"

enum { N = 16 };

static GLuint compile_cs(const char *src)
{
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    GLint ok = 0;
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(sh, sizeof(log), NULL, log);
        fprintf(stderr, "compile failed: %s\n", log);
        glDeleteShader(sh);
        return 0;
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, sh);
    glLinkProgram(prog);
    glDeleteShader(sh);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(prog, sizeof(log), NULL, log);
        fprintf(stderr, "link failed: %s\n", log);
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

int main(void)
{
    GLMContext ctx = createGLMContext(GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV,
                                      GL_DEPTH_COMPONENT, GL_FLOAT, 0, 0);
    if (!ctx) {
        fprintf(stderr, "FAIL: createGLMContext\n");
        return 1;
    }
    if (!CppCreateMGLRendererHeadless(ctx)) {
        fprintf(stderr, "FAIL: CppCreateMGLRendererHeadless\n");
        return 1;
    }
    MGLsetCurrentContext(ctx);
    while (glGetError() != GL_NO_ERROR) {
    }

    GLint max_units = 0, max_combined = 0, max_compute = 0;
    glGetIntegerv(GL_MAX_IMAGE_UNITS, &max_units);
    glGetIntegerv(GL_MAX_COMBINED_IMAGE_UNIFORMS, &max_combined);
    glGetIntegerv(GL_MAX_COMPUTE_IMAGE_UNIFORMS, &max_compute);
    printf("MAX_IMAGE_UNITS=%d MAX_COMBINED_IMAGE_UNIFORMS=%d "
           "MAX_COMPUTE_IMAGE_UNIFORMS=%d\n",
           (int)max_units, (int)max_combined, (int)max_compute);
    if (max_units < N || max_compute < N) {
        fprintf(stderr, "FAIL: advertised caps < %d\n", N);
        return 1;
    }

    static const char *cs =
        "#version 430 core\n"
        "layout(local_size_x = 1) in;\n"
        "layout(r32ui, binding = 0) uniform uimage2D imgs[16];\n"
        "void main() {\n"
        "  for (int i = 0; i < 16; ++i) {\n"
        "    uint v = imageLoad(imgs[i], ivec2(0, 0)).x;\n"
        "    imageStore(imgs[i], ivec2(0, 0), uvec4(v + uint(i) + 1u));\n"
        "  }\n"
        "}\n";
    GLuint prog = compile_cs(cs);
    if (!prog)
        return 1;

    GLuint tex[N];
    glGenTextures(N, tex);
    for (int i = 0; i < N; i++) {
        GLuint init = 100u;
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, 1, 1, 0, GL_RED_INTEGER,
                     GL_UNSIGNED_INT, &init);
        glBindImageTexture((GLuint)i, tex[i], 0, GL_FALSE, 0, GL_READ_WRITE,
                           GL_R32UI);
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        fprintf(stderr, "FAIL: setup GL error 0x%x\n", (unsigned)err);
        return 1;
    }

    glUseProgram(prog);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    glFinish();
    err = glGetError();
    if (err != GL_NO_ERROR) {
        fprintf(stderr, "FAIL: dispatch GL error 0x%x\n", (unsigned)err);
        return 1;
    }

    int fail = 0;
    for (int i = 0; i < N; i++) {
        GLuint got = 0;
        GLuint want = 100u + (GLuint)i + 1u;
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &got);
        if (got != want) {
            fprintf(stderr, "FAIL: imgs[%d]=%u want %u\n", i, got, want);
            fail = 1;
        }
    }
    glDeleteProgram(prog);
    glDeleteTextures(N, tex);
    if (fail)
        return 1;
    printf("PASS: concurrent load+store on %d storage images\n", N);
    return 0;
}
