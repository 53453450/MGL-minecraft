/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Minimal OpenGL ES 3.2 dylib smoke: context + version + DrawArrays.
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

    GLint major = 0, minor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    expect(major == 3 && minor == 2, "ES MAJOR/MINOR 3.2");

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
