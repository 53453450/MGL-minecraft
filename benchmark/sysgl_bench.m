/* sysgl_bench.m — native system OpenGL (4.1 core, NSOpenGLContext) twin of
 * `mgl_benchmark --test triangle`, for MGL-vs-system comparison.
 *
 * Replicates benchmark_triangle_draw() in benchmark/mgl_benchmark.c exactly:
 * same #version 330 core shaders, same VAO/VBO setup, 1000 warm-up draws +
 * glFlush, then 10000 timed glDrawArrays(GL_TRIANGLES, 0, 3) with NO flush or
 * swap inside the timed loop. Reports Draw/s and CPU us/draw.
 *
 * Build:
 *   clang -arch arm64 -O2 -framework Cocoa -framework OpenGL \
 *         -o build/sysgl_bench benchmark/sysgl_bench.m
 */

#import <Cocoa/Cocoa.h>
#import <OpenGL/OpenGL.h>
#import <OpenGL/gl3.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <mach/mach_time.h>

static const char *kVS_src =
    "#version 330 core\n"
    "in vec3 aPos;\n"
    "void main() {\n"
    "    gl_Position = vec4(aPos, 1.0);\n"
    "}\n";

static const char *kFS_src =
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "uniform vec4 uColor;\n"
    "void main() {\n"
    "    FragColor = uColor;\n"
    "}\n";

static GLuint build_program(const char *vs, const char *fs)
{
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &vs, NULL);
    glCompileShader(v);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fs, NULL);
    glCompileShader(f);
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    GLint ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        fprintf(stderr, "link failed: %s\n", log);
        exit(1);
    }
    return p;
}

static uint64_t now_ns(void)
{
    return (uint64_t)mach_absolute_time();
}

static double ns_to_ms(uint64_t ns)
{
    static double scale = 0.0;
    if (scale == 0.0) {
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        scale = (double)tb.numer / (double)tb.denom;
    }
    return (double)ns * scale;
}

static void benchmark_triangle_draw(void)
{
    GLuint prog = build_program(kVS_src, kFS_src);
    glUseProgram(prog);
    GLint loc = glGetUniformLocation(prog, "uColor");
    glUniform4f(loc, 1.0f, 0.0f, 0.0f, 1.0f);

    float verts[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f,
    };

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                          (void *)0);

    for (int i = 0; i < 1000; i++)
        glDrawArrays(GL_TRIANGLES, 0, 3);
    glFlush();

    /* 5 rounds, same as mgl_benchmark's single measurement repeated */
    const int N = 10000;
    for (int r = 0; r < 5; r++) {
        uint64_t start = now_ns();
        for (int i = 0; i < N; i++)
            glDrawArrays(GL_TRIANGLES, 0, 3);
        uint64_t elapsed = now_ns() - start;
        glFlush();

        double ns = ns_to_ms(elapsed);
        double draws_per_s = (double)N / (ns / 1e9);
        double us_per_draw = ns / 1000.0 / (double)N;
        printf("Triangle Draw  Draw/s       %.0f\n", draws_per_s);
        printf("Triangle Draw  CPU us/draw  %.2f\n", us_per_draw);
    }

    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
}

int main(int argc, char **argv)
{
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        (void)app;

        NSOpenGLPixelFormatAttribute attrs[] = {
            NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion4_1Core,
            NSOpenGLPFADoubleBuffer,
            0
        };
        NSOpenGLPixelFormat *pf =
            [[NSOpenGLPixelFormat alloc] initWithAttributes:attrs];
        if (!pf) { fprintf(stderr, "no 4.1 core pixel format\n"); return 1; }
        NSOpenGLContext *ctx =
            [[NSOpenGLContext alloc] initWithFormat:pf shareContext:nil];
        if (!ctx) { fprintf(stderr, "context creation failed\n"); return 1; }

        /* tiny invisible window to give the context a real surface */
        NSRect frame = NSMakeRect(0, 0, 640, 480);
        NSWindow *win = [[NSWindow alloc]
            initWithContentRect:frame
                      styleMask:NSWindowStyleMaskTitled
                        backing:NSBackingStoreBuffered
                          defer:NO];
        [win orderOut:nil];   /* never shown */
        [ctx setView:[win contentView]];
        [ctx makeCurrentContext];

        const GLubyte *ver = glGetString(GL_VERSION);
        const GLubyte *ren = glGetString(GL_RENDERER);
        printf("GL_VERSION: %s\nGL_RENDERER: %s\n", ver, ren);

        glViewport(0, 0, 640, 480);
        benchmark_triangle_draw();
        return 0;
    }
}
