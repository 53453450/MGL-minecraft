/*
 * mgl_benchmark.c
 *
 * Comprehensive benchmark program for the MGL (OpenGL -> Metal translation
 * layer) project.  Creates a hidden GLFW window, loads GL function pointers
 * through glfwGetProcAddress (so every call is routed through MGL), and runs
 * 10 benchmark categories that measure the translation overhead.
 *
 * Build:
 *   cc -D-Wall -gfull -O2 -arch arm64 \
 *       -I./external/glfw/include \
 *       -IMGL/include -IMGL/include/GL \
 *       -DMGL_GL_CORE \
 *       -isysroot $(xcrun --sdk macosx --show-sdk-path) \
 *       benchmark/mgl_benchmark.c \
 *       -Lbuild -lmgl -lglfw \
 *       -framework Cocoa -framework CoreFoundation -framework CoreGraphics \
 *       -framework IOKit -framework Foundation -framework QuartzCore \
 *       -framework Metal -framework OpenGL \
 *       -o build/mgl_benchmark
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <math.h>

#include <mach/mach_time.h>

/* Prevent GLFW from pulling in the system OpenGL headers — we want MGL's
 * glcorearb.h to be the sole source of GL type/enum definitions so that the
 * function pointers we load via glfwGetProcAddress match MGL's signatures. */
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

/* glcorearb.h provides the GLenum/GLuint typedefs, the GL_* enum constants,
 * and the PFNGL* function-pointer typedefs.  It also declares the GLAPI
 * prototypes, but we never call those directly — every GL entry point is
 * resolved at runtime through glfwGetProcAddress so the calls go through MGL,
 * not the system OpenGL framework. */
#define GL_GLEXT_PROTOTYPES 1
#include <GL/glcorearb.h>

/* ======================================================================== */
/*  GL function-pointer declarations                                         */
/* ======================================================================== */

/* Each GL entry point is stored in a function pointer loaded at runtime via
 * glfwGetProcAddress so that every call is routed through MGL.  The PFNGL*
 * typedefs come from glcorearb.h. */
static PFNGLGETERRORPROC             p_glGetError            = NULL;
static PFNGLGETSTRINGPROC            p_glGetString           = NULL;
static PFNGLCLEARCOLORPROC           p_glClearColor          = NULL;
static PFNGLCLEARPROC                p_glClear               = NULL;
static PFNGLVIEWPORTPROC             p_glViewport            = NULL;
static PFNGLENABLEPROC               p_glEnable              = NULL;
static PFNGLDISABLEPROC              p_glDisable             = NULL;
static PFNGLBLENDFUNCPROC            p_glBlendFunc           = NULL;
static PFNGLFLUSHPROC                p_glFlush               = NULL;
static PFNGLFINISHPROC               p_glFinish              = NULL;

static PFNGLGENVERTEXARRAYSPROC      p_glGenVertexArrays     = NULL;
static PFNGLBINDVERTEXARRAYPROC      p_glBindVertexArray     = NULL;
static PFNGLDELETEVERTEXARRAYSPROC   p_glDeleteVertexArrays  = NULL;
static PFNGLGENBUFFERSPROC           p_glGenBuffers          = NULL;
static PFNGLBINDBUFFERPROC           p_glBindBuffer          = NULL;
static PFNGLBUFFERDATAPROC           p_glBufferData          = NULL;
static PFNGLBUFFERSUBDATAPROC        p_glBufferSubData       = NULL;
static PFNGLDELETEBUFFERSPROC        p_glDeleteBuffers       = NULL;
static PFNGLENABLEVERTEXATTRIBARRAYPROC p_glEnableVertexAttribArray = NULL;
static PFNGLVERTEXATTRIBPOINTERPROC  p_glVertexAttribPointer = NULL;

static PFNGLGENTEXTURESPROC          p_glGenTextures         = NULL;
static PFNGLBINDTEXTUREPROC          p_glBindTexture         = NULL;
static PFNGLDELETETEXTURESPROC       p_glDeleteTextures      = NULL;
static PFNGLTEXIMAGE2DPROC           p_glTexImage2D          = NULL;
static PFNGLTEXSUBIMAGE2DPROC        p_glTexSubImage2D       = NULL;
static PFNGLTEXPARAMETERIPROC        p_glTexParameteri       = NULL;
static PFNGLACTIVETEXTUREPROC        p_glActiveTexture       = NULL;

static PFNGLCREATEPROGRAMPROC        p_glCreateProgram       = NULL;
static PFNGLCREATESHADERPROC         p_glCreateShader        = NULL;
static PFNGLSHADERSOURCEPROC         p_glShaderSource        = NULL;
static PFNGLCOMPILESHADERPROC        p_glCompileShader       = NULL;
static PFNGLATTACHSHADERPROC         p_glAttachShader        = NULL;
static PFNGLLINKPROGRAMPROC          p_glLinkProgram         = NULL;
static PFNGLUSEPROGRAMPROC           p_glUseProgram          = NULL;
static PFNGLDELETEPROGRAMPROC        p_glDeleteProgram       = NULL;
static PFNGLDELETESHADERPROC         p_glDeleteShader        = NULL;
static PFNGLGETPROGRAMIVPROC         p_glGetProgramiv        = NULL;
static PFNGLGETSHADERIVPROC          p_glGetShaderiv         = NULL;
static PFNGLGETUNIFORMLOCATIONPROC   p_glGetUniformLocation  = NULL;
static PFNGLUNIFORM4FPROC            p_glUniform4f           = NULL;
static PFNGLUNIFORMMATRIX4FVPROC     p_glUniformMatrix4fv    = NULL;

static PFNGLGENFRAMEBUFFERSPROC      p_glGenFramebuffers     = NULL;
static PFNGLBINDFRAMEBUFFERPROC      p_glBindFramebuffer     = NULL;
static PFNGLDELETEFRAMEBUFFERSPROC   p_glDeleteFramebuffers  = NULL;
static PFNGLFRAMEBUFFERTEXTURE2DPROC p_glFramebufferTexture2D = NULL;
static PFNGLCHECKFRAMEBUFFERSTATUSPROC p_glCheckFramebufferStatus = NULL;

static PFNGLDRAWARRAYSPROC           p_glDrawArrays          = NULL;
static PFNGLDRAWARRAYSINSTANCEDPROC  p_glDrawArraysInstanced = NULL;

static PFNGLGENQUERIESPROC           p_glGenQueries          = NULL;
static PFNGLDELETEQUERIESPROC        p_glDeleteQueries       = NULL;
static PFNGLBEGINQUERYPROC           p_glBeginQuery          = NULL;
static PFNGLENDQUERYPROC             p_glEndQuery            = NULL;
static PFNGLQUERYCOUNTERPROC         p_glQueryCounter        = NULL;
static PFNGLGETQUERYOBJECTUI64VPROC  p_glGetQueryObjectui64v = NULL;

/* API dispatch benchmark — additional no-op bind / query entry points.
 * (glUseProgram, glBindTexture, glActiveTexture, glEnableVertexAttribArray,
 *  glBindVertexArray, glBindBuffer are already declared above.) */
static PFNGLSCISSORPROC              p_glScissor             = NULL;
static PFNGLDEPTHMASKPROC            p_glDepthMask           = NULL;
static PFNGLCOLORMASKPROC            p_glColorMask           = NULL;
static PFNGLCULLFACEPROC             p_glCullFace            = NULL;
static PFNGLPOLYGONMODEPROC          p_glPolygonMode         = NULL;
static PFNGLPIXELSTOREIPROC          p_glPixelStorei         = NULL;
static PFNGLGETINTEGERVPROC          p_glGetIntegerv         = NULL;
static PFNGLDISABLEVERTEXATTRIBARRAYPROC p_glDisableVertexAttribArray = NULL;
static PFNGLDRAWELEMENTSPROC         p_glDrawElements        = NULL;

/* ======================================================================== */
/*  Timing utility (mach_absolute_time)                                      */
/* ======================================================================== */

static mach_timebase_info_data_t g_timebase;

static void init_timing(void)
{
    mach_timebase_info(&g_timebase);
}

/* Returns current monotonic time in nanoseconds. */
static uint64_t now_ns(void)
{
    uint64_t t = mach_absolute_time();
    return (uint64_t)((double)t * (double)g_timebase.numer /
                      (double)g_timebase.denom);
}

/* ======================================================================== */
/*  GL helper utilities                                                      */
/* ======================================================================== */

static void check_gl_error(const char *context)
{
    GLenum err = p_glGetError();
    if (err != GL_NO_ERROR) {
        fprintf(stderr, "  [warning] GL error 0x%04X during %s\n",
                (unsigned)err, context);
    }
}

static int load_gl(void)
{
#define LOAD_GL(ptr, type, name)                                         \
    do {                                                                 \
        ptr = (type)glfwGetProcAddress(name);                           \
        if (!ptr) {                                                     \
            fprintf(stderr, "Failed to load GL function: %s\n", name);  \
            return 0;                                                    \
        }                                                                \
    } while (0)

    LOAD_GL(p_glGetError,            PFNGLGETERRORPROC,             "glGetError");
    LOAD_GL(p_glGetString,           PFNGLGETSTRINGPROC,            "glGetString");
    LOAD_GL(p_glClearColor,          PFNGLCLEARCOLORPROC,           "glClearColor");
    LOAD_GL(p_glClear,               PFNGLCLEARPROC,                "glClear");
    LOAD_GL(p_glViewport,            PFNGLVIEWPORTPROC,             "glViewport");
    LOAD_GL(p_glEnable,              PFNGLENABLEPROC,               "glEnable");
    LOAD_GL(p_glDisable,             PFNGLDISABLEPROC,              "glDisable");
    LOAD_GL(p_glBlendFunc,           PFNGLBLENDFUNCPROC,            "glBlendFunc");
    LOAD_GL(p_glFlush,               PFNGLFLUSHPROC,                "glFlush");
    LOAD_GL(p_glFinish,              PFNGLFINISHPROC,               "glFinish");

    LOAD_GL(p_glGenVertexArrays,     PFNGLGENVERTEXARRAYSPROC,      "glGenVertexArrays");
    LOAD_GL(p_glBindVertexArray,     PFNGLBINDVERTEXARRAYPROC,      "glBindVertexArray");
    LOAD_GL(p_glDeleteVertexArrays,  PFNGLDELETEVERTEXARRAYSPROC,   "glDeleteVertexArrays");
    LOAD_GL(p_glGenBuffers,          PFNGLGENBUFFERSPROC,           "glGenBuffers");
    LOAD_GL(p_glBindBuffer,          PFNGLBINDBUFFERPROC,           "glBindBuffer");
    LOAD_GL(p_glBufferData,          PFNGLBUFFERDATAPROC,           "glBufferData");
    LOAD_GL(p_glBufferSubData,       PFNGLBUFFERSUBDATAPROC,        "glBufferSubData");
    LOAD_GL(p_glDeleteBuffers,       PFNGLDELETEBUFFERSPROC,        "glDeleteBuffers");
    LOAD_GL(p_glEnableVertexAttribArray, PFNGLENABLEVERTEXATTRIBARRAYPROC, "glEnableVertexAttribArray");
    LOAD_GL(p_glVertexAttribPointer, PFNGLVERTEXATTRIBPOINTERPROC,  "glVertexAttribPointer");

    LOAD_GL(p_glGenTextures,         PFNGLGENTEXTURESPROC,          "glGenTextures");
    LOAD_GL(p_glBindTexture,         PFNGLBINDTEXTUREPROC,          "glBindTexture");
    LOAD_GL(p_glDeleteTextures,      PFNGLDELETETEXTURESPROC,       "glDeleteTextures");
    LOAD_GL(p_glTexImage2D,          PFNGLTEXIMAGE2DPROC,           "glTexImage2D");
    LOAD_GL(p_glTexSubImage2D,       PFNGLTEXSUBIMAGE2DPROC,        "glTexSubImage2D");
    LOAD_GL(p_glTexParameteri,       PFNGLTEXPARAMETERIPROC,        "glTexParameteri");
    LOAD_GL(p_glActiveTexture,       PFNGLACTIVETEXTUREPROC,        "glActiveTexture");

    LOAD_GL(p_glCreateProgram,       PFNGLCREATEPROGRAMPROC,        "glCreateProgram");
    LOAD_GL(p_glCreateShader,        PFNGLCREATESHADERPROC,         "glCreateShader");
    LOAD_GL(p_glShaderSource,        PFNGLSHADERSOURCEPROC,         "glShaderSource");
    LOAD_GL(p_glCompileShader,       PFNGLCOMPILESHADERPROC,        "glCompileShader");
    LOAD_GL(p_glAttachShader,        PFNGLATTACHSHADERPROC,         "glAttachShader");
    LOAD_GL(p_glLinkProgram,         PFNGLLINKPROGRAMPROC,          "glLinkProgram");
    LOAD_GL(p_glUseProgram,          PFNGLUSEPROGRAMPROC,           "glUseProgram");
    LOAD_GL(p_glDeleteProgram,       PFNGLDELETEPROGRAMPROC,        "glDeleteProgram");
    LOAD_GL(p_glDeleteShader,        PFNGLDELETESHADERPROC,         "glDeleteShader");
    LOAD_GL(p_glGetProgramiv,        PFNGLGETPROGRAMIVPROC,         "glGetProgramiv");
    LOAD_GL(p_glGetShaderiv,         PFNGLGETSHADERIVPROC,          "glGetShaderiv");
    LOAD_GL(p_glGetUniformLocation,  PFNGLGETUNIFORMLOCATIONPROC,   "glGetUniformLocation");
    LOAD_GL(p_glUniform4f,           PFNGLUNIFORM4FPROC,            "glUniform4f");
    LOAD_GL(p_glUniformMatrix4fv,    PFNGLUNIFORMMATRIX4FVPROC,     "glUniformMatrix4fv");

    LOAD_GL(p_glGenFramebuffers,     PFNGLGENFRAMEBUFFERSPROC,      "glGenFramebuffers");
    LOAD_GL(p_glBindFramebuffer,     PFNGLBINDFRAMEBUFFERPROC,      "glBindFramebuffer");
    LOAD_GL(p_glDeleteFramebuffers,  PFNGLDELETEFRAMEBUFFERSPROC,   "glDeleteFramebuffers");
    LOAD_GL(p_glFramebufferTexture2D, PFNGLFRAMEBUFFERTEXTURE2DPROC, "glFramebufferTexture2D");
    LOAD_GL(p_glCheckFramebufferStatus, PFNGLCHECKFRAMEBUFFERSTATUSPROC, "glCheckFramebufferStatus");

    LOAD_GL(p_glDrawArrays,          PFNGLDRAWARRAYSPROC,           "glDrawArrays");
    LOAD_GL(p_glDrawArraysInstanced, PFNGLDRAWARRAYSINSTANCEDPROC,  "glDrawArraysInstanced");

    LOAD_GL(p_glGenQueries,          PFNGLGENQUERIESPROC,           "glGenQueries");
    LOAD_GL(p_glDeleteQueries,       PFNGLDELETEQUERIESPROC,        "glDeleteQueries");
    LOAD_GL(p_glBeginQuery,          PFNGLBEGINQUERYPROC,           "glBeginQuery");
    LOAD_GL(p_glEndQuery,            PFNGLENDQUERYPROC,             "glEndQuery");
    LOAD_GL(p_glQueryCounter,        PFNGLQUERYCOUNTERPROC,         "glQueryCounter");
    LOAD_GL(p_glGetQueryObjectui64v, PFNGLGETQUERYOBJECTUI64VPROC,  "glGetQueryObjectui64v");

    LOAD_GL(p_glScissor,              PFNGLSCISSORPROC,              "glScissor");
    LOAD_GL(p_glDepthMask,            PFNGLDEPTHMASKPROC,            "glDepthMask");
    LOAD_GL(p_glColorMask,            PFNGLCOLORMASKPROC,            "glColorMask");
    LOAD_GL(p_glCullFace,             PFNGLCULLFACEPROC,             "glCullFace");
    LOAD_GL(p_glPolygonMode,          PFNGLPOLYGONMODEPROC,          "glPolygonMode");
    LOAD_GL(p_glPixelStorei,          PFNGLPIXELSTOREIPROC,          "glPixelStorei");
    LOAD_GL(p_glGetIntegerv,          PFNGLGETINTEGERVPROC,          "glGetIntegerv");
    LOAD_GL(p_glDisableVertexAttribArray, PFNGLDISABLEVERTEXATTRIBARRAYPROC, "glDisableVertexAttribArray");
    LOAD_GL(p_glDrawElements,         PFNGLDRAWELEMENTSPROC,         "glDrawElements");

#undef LOAD_GL
    return 1;
}

static GLuint compile_shader(GLenum type, const char *src)
{
    GLuint sh = p_glCreateShader(type);
    p_glShaderSource(sh, 1, (const GLchar *const *)&src, NULL);
    p_glCompileShader(sh);
    GLint status = GL_FALSE;
    p_glGetShaderiv(sh, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        fprintf(stderr, "  [warning] shader compile failed (type 0x%X)\n",
                (unsigned)type);
    }
    return sh;
}

static GLuint build_program(const char *vs_src, const char *fs_src)
{
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
    GLuint prog = p_glCreateProgram();
    p_glAttachShader(prog, vs);
    p_glAttachShader(prog, fs);
    p_glLinkProgram(prog);
    GLint status = GL_FALSE;
    p_glGetProgramiv(prog, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
        fprintf(stderr, "  [warning] program link failed\n");
    }
    p_glDeleteShader(vs);
    p_glDeleteShader(fs);
    return prog;
}

/* Simple solid-color shaders. */
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

/* A second pair of shaders with slightly different code so the Metal pipeline
 * state actually differs (used by the pipeline-switch benchmark). */
static const char *kVS2_src =
    "#version 330 core\n"
    "in vec3 aPos;\n"
    "uniform mat4 uMVP;\n"
    "void main() {\n"
    "    gl_Position = uMVP * vec4(aPos, 1.0);\n"
    "}\n";

static const char *kFS2_src =
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "uniform vec4 uColor;\n"
    "void main() {\n"
    "    FragColor = vec4(uColor.rgb * 0.5, uColor.a);\n"
    "}\n";

/* ======================================================================== */
/*  Output helpers                                                           */
/* ======================================================================== */

static void print_table_header(void)
{
    printf("========================================\n");
    printf("MGL Benchmark Results\n");
    printf("========================================\n");
    printf("%-22s   %-19s   %s\n", "Test", "Metric", "Result");
    printf("%-22s   %-19s   %s\n",
           "--------------------", "-------------------", "--------");
}

static void print_table_footer(void)
{
    printf("========================================\n");
}

static void print_row(const char *test, const char *metric,
                      const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    printf("%-22s   %-19s   ", test, metric);
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
}

/* ======================================================================== */
/*  Global window handle (needed by swap-based benchmarks)                   */
/* ======================================================================== */

static GLFWwindow *g_window = NULL;

/* ======================================================================== */
/*  Benchmark 1: Empty Draw                                                  */
/*  glDrawArrays(GL_TRIANGLES, 0, 0) with an empty VAO.                     */
/* ======================================================================== */

static void benchmark_empty_draw(void)
{
    GLuint vao;
    p_glGenVertexArrays(1, &vao);
    p_glBindVertexArray(vao);

    /* Warm up. */
    for (int i = 0; i < 1000; i++) {
        p_glDrawArrays(GL_TRIANGLES, 0, 0);
    }
    p_glFlush();

    const int N = 10000;
    uint64_t start = now_ns();
    for (int i = 0; i < N; i++) {
        p_glDrawArrays(GL_TRIANGLES, 0, 0);
    }
    uint64_t elapsed = now_ns() - start;

    double ns_per_draw = (double)elapsed / (double)N;
    print_row("Empty Draw", "ns/Draw", "%.1f", ns_per_draw);

    p_glBindVertexArray(0);
    p_glDeleteVertexArrays(1, &vao);
}

/* ======================================================================== */
/*  Benchmark 2: Triangle Draw                                               */
/*  Single triangle with position attribute, solid-color shader.            */
/* ======================================================================== */

static void benchmark_triangle_draw(void)
{
    GLuint prog = build_program(kVS_src, kFS_src);
    p_glUseProgram(prog);
    GLint loc = p_glGetUniformLocation(prog, "uColor");
    p_glUniform4f(loc, 1.0f, 0.0f, 0.0f, 1.0f);

    float verts[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f,
    };

    GLuint vao, vbo;
    p_glGenVertexArrays(1, &vao);
    p_glBindVertexArray(vao);
    p_glGenBuffers(1, &vbo);
    p_glBindBuffer(GL_ARRAY_BUFFER, vbo);
    p_glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    p_glEnableVertexAttribArray(0);
    p_glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                            (void *)0);

    /* Warm up. */
    for (int i = 0; i < 1000; i++) {
        p_glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    p_glFlush();

    const int N = 10000;
    uint64_t start = now_ns();
    for (int i = 0; i < N; i++) {
        p_glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    uint64_t elapsed = now_ns() - start;

    double draws_per_s = (double)N / ((double)elapsed / 1e9);
    double us_per_draw = (double)elapsed / 1000.0 / (double)N;

    print_row("Triangle Draw", "Draw/s", "%.0f", draws_per_s);
    print_row("Triangle Draw", "CPU us/draw", "%.2f", us_per_draw);

    p_glDeleteBuffers(1, &vbo);
    p_glDeleteVertexArrays(1, &vao);
    p_glDeleteProgram(prog);
}

/* ======================================================================== */
/*  Benchmark 3: Batch Draw (deferred vs immediate)                          */
/*  Compares calling glDrawArrays 10000x with and without glFlush.          */
/* ======================================================================== */

static void benchmark_batch_draw(void)
{
    GLuint prog = build_program(kVS_src, kFS_src);
    p_glUseProgram(prog);
    GLint loc = p_glGetUniformLocation(prog, "uColor");
    p_glUniform4f(loc, 0.0f, 1.0f, 0.0f, 1.0f);

    float verts[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f,
    };

    GLuint vao, vbo;
    p_glGenVertexArrays(1, &vao);
    p_glBindVertexArray(vao);
    p_glGenBuffers(1, &vbo);
    p_glBindBuffer(GL_ARRAY_BUFFER, vbo);
    p_glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    p_glEnableVertexAttribArray(0);
    p_glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                            (void *)0);

    const int N = 10000;

    /* --- Batched (deferred): MGL batches internally, no flush. --- */
    for (int i = 0; i < 1000; i++) {
        p_glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    p_glFlush();

    uint64_t start = now_ns();
    for (int i = 0; i < N; i++) {
        p_glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    p_glFlush();
    uint64_t elapsed_batched = now_ns() - start;
    double batched_dps = (double)N / ((double)elapsed_batched / 1e9);

    /* --- Immediate: flush after every draw to force submission. --- */
    for (int i = 0; i < 1000; i++) {
        p_glDrawArrays(GL_TRIANGLES, 0, 3);
        p_glFlush();
    }

    start = now_ns();
    for (int i = 0; i < N; i++) {
        p_glDrawArrays(GL_TRIANGLES, 0, 3);
        p_glFlush();
    }
    uint64_t elapsed_immediate = now_ns() - start;
    double immediate_dps = (double)N / ((double)elapsed_immediate / 1e9);

    print_row("Batch Draw (deferred)", "Draw/s", "%.0f", batched_dps);
    print_row("Batch Draw (immediate)", "Draw/s", "%.0f", immediate_dps);

    p_glDeleteBuffers(1, &vbo);
    p_glDeleteVertexArrays(1, &vao);
    p_glDeleteProgram(prog);
}

/* ======================================================================== */
/*  Benchmark 4: Texture Upload                                              */
/*  1024x1024 RGBA8 via glTexSubImage2D.                                    */
/* ======================================================================== */

static void benchmark_texture_upload(void)
{
    const int W = 1024, H = 1024;
    const size_t data_size = (size_t)W * H * 4; /* 4 MB */

    GLuint tex;
    p_glGenTextures(1, &tex);
    p_glBindTexture(GL_TEXTURE_2D, tex);
    p_glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, NULL);
    p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    uint8_t *pixels = (uint8_t *)malloc(data_size);
    if (!pixels) {
        fprintf(stderr, "  [warning] alloc failed in texture upload\n");
        p_glDeleteTextures(1, &tex);
        return;
    }
    memset(pixels, 0xAB, data_size);

    /* Warm up. */
    for (int i = 0; i < 5; i++) {
        p_glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA,
                          GL_UNSIGNED_BYTE, pixels);
    }
    p_glFlush();

    const int N = 100;
    uint64_t start = now_ns();
    for (int i = 0; i < N; i++) {
        p_glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA,
                          GL_UNSIGNED_BYTE, pixels);
    }
    p_glFlush();
    uint64_t elapsed = now_ns() - start;

    double secs = (double)elapsed / 1e9;
    double total_mb = (double)N * (double)data_size / (1024.0 * 1024.0);
    double mb_per_s = total_mb / secs;
    double us_per_call = (double)elapsed / 1000.0 / (double)N;

    print_row("Texture Upload", "MB/s", "%.1f", mb_per_s);
    print_row("Texture Upload", "us/call", "%.1f", us_per_call);

    free(pixels);
    p_glDeleteTextures(1, &tex);
}

/* ======================================================================== */
/*  Benchmark 5: Buffer Upload                                               */
/*  1 MB GL_ARRAY_BUFFER via glBufferSubData.                               */
/* ======================================================================== */

static void benchmark_buffer_upload(void)
{
    const size_t data_size = 1024 * 1024; /* 1 MB */

    GLuint buf;
    p_glGenBuffers(1, &buf);
    p_glBindBuffer(GL_ARRAY_BUFFER, buf);
    p_glBufferData(GL_ARRAY_BUFFER, data_size, NULL, GL_DYNAMIC_DRAW);

    uint8_t *data = (uint8_t *)malloc(data_size);
    if (!data) {
        fprintf(stderr, "  [warning] alloc failed in buffer upload\n");
        p_glDeleteBuffers(1, &buf);
        return;
    }
    memset(data, 0xCD, data_size);

    /* Warm up. */
    for (int i = 0; i < 5; i++) {
        p_glBufferSubData(GL_ARRAY_BUFFER, 0, data_size, data);
    }
    p_glFlush();

    const int N = 100;
    uint64_t start = now_ns();
    for (int i = 0; i < N; i++) {
        p_glBufferSubData(GL_ARRAY_BUFFER, 0, data_size, data);
    }
    p_glFlush();
    uint64_t elapsed = now_ns() - start;

    double secs = (double)elapsed / 1e9;
    double total_mb = (double)N * (double)data_size / (1024.0 * 1024.0);
    double mb_per_s = total_mb / secs;

    print_row("Buffer Upload", "MB/s", "%.1f", mb_per_s);

    free(data);
    p_glDeleteBuffers(1, &buf);
}

/* ======================================================================== */
/*  Benchmark 6: State Changes                                               */
/*  Toggle glEnable/glDisable and glBlendFunc.                              */
/* ======================================================================== */

static void benchmark_state_changes(void)
{
    const int N = 10000;

    /* --- Depth test toggle --- */
    for (int i = 0; i < 1000; i++) {
        p_glEnable(GL_DEPTH_TEST);
        p_glDisable(GL_DEPTH_TEST);
    }

    uint64_t start = now_ns();
    for (int i = 0; i < N; i++) {
        p_glEnable(GL_DEPTH_TEST);
        p_glDisable(GL_DEPTH_TEST);
    }
    uint64_t elapsed = now_ns() - start;
    /* Two calls per iteration. */
    double ns_per_call = (double)elapsed / (double)(N * 2);
    print_row("State: Depth Toggle", "ns/call", "%.1f", ns_per_call);

    /* --- Blend func toggle --- */
    for (int i = 0; i < 1000; i++) {
        p_glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        p_glBlendFunc(GL_ONE, GL_ZERO);
    }

    start = now_ns();
    for (int i = 0; i < N; i++) {
        p_glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        p_glBlendFunc(GL_ONE, GL_ZERO);
    }
    elapsed = now_ns() - start;
    ns_per_call = (double)elapsed / (double)(N * 2);
    print_row("State: BlendFunc Toggle", "ns/call", "%.1f", ns_per_call);
}

/* ======================================================================== */
/*  Benchmark 7: Pipeline Switch                                             */
/*  Alternate two programs, two VAOs (different formats), two FBOs.         */
/* ======================================================================== */

static void benchmark_pipeline_switch(void)
{
    /* Two different programs. */
    GLuint progA = build_program(kVS_src, kFS_src);
    GLuint progB = build_program(kVS2_src, kFS2_src);

    p_glUseProgram(progA);
    GLint locA = p_glGetUniformLocation(progA, "uColor");
    p_glUniform4f(locA, 1.0f, 0.0f, 0.0f, 1.0f);

    p_glUseProgram(progB);
    GLint locB_color = p_glGetUniformLocation(progB, "uColor");
    GLint locB_mvp = p_glGetUniformLocation(progB, "uMVP");
    p_glUniform4f(locB_color, 0.0f, 0.0f, 1.0f, 1.0f);
    float mvp[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    p_glUniformMatrix4fv(locB_mvp, 1, GL_FALSE, mvp);

    /* VAO A: position only (vec3 at location 0). */
    float vertsA[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f,
    };
    GLuint vaoA, vboA;
    p_glGenVertexArrays(1, &vaoA);
    p_glBindVertexArray(vaoA);
    p_glGenBuffers(1, &vboA);
    p_glBindBuffer(GL_ARRAY_BUFFER, vboA);
    p_glBufferData(GL_ARRAY_BUFFER, sizeof(vertsA), vertsA, GL_STATIC_DRAW);
    p_glEnableVertexAttribArray(0);
    p_glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                            (void *)0);

    /* VAO B: position (vec3) + texcoord (vec2). */
    float vertsB[] = {
        -0.5f, -0.5f, 0.0f, 0.0f, 0.0f,
         0.5f, -0.5f, 0.0f, 1.0f, 0.0f,
         0.0f,  0.5f, 0.0f, 0.5f, 1.0f,
    };
    GLuint vaoB, vboB;
    p_glGenVertexArrays(1, &vaoB);
    p_glBindVertexArray(vaoB);
    p_glGenBuffers(1, &vboB);
    p_glBindBuffer(GL_ARRAY_BUFFER, vboB);
    p_glBufferData(GL_ARRAY_BUFFER, sizeof(vertsB), vertsB, GL_STATIC_DRAW);
    p_glEnableVertexAttribArray(0);
    p_glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                            (void *)0);
    p_glEnableVertexAttribArray(1);
    p_glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                            (void *)(3 * sizeof(float)));

    /* Two FBOs with different-sized color textures. */
    GLuint texA, texB, fboA, fboB;
    p_glGenTextures(1, &texA);
    p_glBindTexture(GL_TEXTURE_2D, texA);
    p_glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 256, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, NULL);
    p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    p_glGenTextures(1, &texB);
    p_glBindTexture(GL_TEXTURE_2D, texB);
    p_glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 512, 512, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, NULL);
    p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    p_glGenFramebuffers(1, &fboA);
    p_glBindFramebuffer(GL_FRAMEBUFFER, fboA);
    p_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, texA, 0);

    p_glGenFramebuffers(1, &fboB);
    p_glBindFramebuffer(GL_FRAMEBUFFER, fboB);
    p_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, texB, 0);

    /* Warm up. */
    for (int i = 0; i < 100; i++) {
        p_glUseProgram(progA);
        p_glBindVertexArray(vaoA);
        p_glBindFramebuffer(GL_FRAMEBUFFER, fboA);
        p_glUseProgram(progB);
        p_glBindVertexArray(vaoB);
        p_glBindFramebuffer(GL_FRAMEBUFFER, fboB);
    }
    p_glFlush();

    const int N = 10000;
    /* Each iteration performs two full switches: A and B. */
    uint64_t start = now_ns();
    for (int i = 0; i < N; i++) {
        p_glUseProgram(progA);
        p_glBindVertexArray(vaoA);
        p_glBindFramebuffer(GL_FRAMEBUFFER, fboA);

        p_glUseProgram(progB);
        p_glBindVertexArray(vaoB);
        p_glBindFramebuffer(GL_FRAMEBUFFER, fboB);
    }
    p_glFlush();
    uint64_t elapsed = now_ns() - start;

    double ns_per_switch = (double)elapsed / (double)(N * 2);
    print_row("Pipeline Switch", "ns/call", "%.1f", ns_per_switch);

    /* Restore default framebuffer. */
    p_glBindFramebuffer(GL_FRAMEBUFFER, 0);
    p_glBindVertexArray(0);
    p_glUseProgram(0);

    p_glDeleteFramebuffers(1, &fboA);
    p_glDeleteFramebuffers(1, &fboB);
    p_glDeleteTextures(1, &texA);
    p_glDeleteTextures(1, &texB);
    p_glDeleteBuffers(1, &vboA);
    p_glDeleteBuffers(1, &vboB);
    p_glDeleteVertexArrays(1, &vaoA);
    p_glDeleteVertexArrays(1, &vaoB);
    p_glDeleteProgram(progA);
    p_glDeleteProgram(progB);
}

/* ======================================================================== */
/*  Benchmark 8: Uniform Update                                              */
/*  glUniform4f and glUniformMatrix4fv.                                     */
/* ======================================================================== */

static void benchmark_uniform_update(void)
{
    GLuint prog = build_program(kVS2_src, kFS2_src);
    p_glUseProgram(prog);

    GLint loc_vec4 = p_glGetUniformLocation(prog, "uColor");
    GLint loc_mat4 = p_glGetUniformLocation(prog, "uMVP");

    float matrix[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    const int N = 10000;

    /* --- vec4 uniform --- */
    for (int i = 0; i < 1000; i++) {
        p_glUniform4f(loc_vec4, 1.0f, 0.5f, 0.25f, 1.0f);
    }

    uint64_t start = now_ns();
    for (int i = 0; i < N; i++) {
        p_glUniform4f(loc_vec4, 1.0f, 0.5f, 0.25f, 1.0f);
    }
    uint64_t elapsed = now_ns() - start;
    double ns_per_call = (double)elapsed / (double)N;
    print_row("Uniform: glUniform4f", "ns/call", "%.1f", ns_per_call);

    /* --- mat4 uniform --- */
    for (int i = 0; i < 1000; i++) {
        p_glUniformMatrix4fv(loc_mat4, 1, GL_FALSE, matrix);
    }

    start = now_ns();
    for (int i = 0; i < N; i++) {
        p_glUniformMatrix4fv(loc_mat4, 1, GL_FALSE, matrix);
    }
    elapsed = now_ns() - start;
    ns_per_call = (double)elapsed / (double)N;
    print_row("Uniform: glUniformMatrix4fv", "ns/call", "%.1f", ns_per_call);

    p_glDeleteProgram(prog);
}

/* ======================================================================== */
/*  Benchmark 9: GPU Time                                                    */
/*  Separates CPU submit time from GPU execution time.                      */
/*  MGL's GL_TIMESTAMP/GL_TIME_ELAPSED queries are stub implementations    */
/*  (return fake counters), so we use a glFinish()-based approach:          */
/*    - CPU-only time: wall-clock around the draw loop (no finish)          */
/*    - CPU+GPU time: wall-clock around draw loop + glFinish()              */
/*    - GPU time = (CPU+GPU) - CPU                                          */
/* ======================================================================== */

static void benchmark_gpu_time(void)
{
    GLuint prog = build_program(kVS_src, kFS_src);
    p_glUseProgram(prog);
    GLint loc = p_glGetUniformLocation(prog, "uColor");
    p_glUniform4f(loc, 0.2f, 0.4f, 0.8f, 1.0f);

    /* 1000 triangles = 3000 vertices. */
    const int tri_count = 1000;
    const int vert_count = tri_count * 3;
    float *verts = (float *)malloc((size_t)vert_count * 3 * sizeof(float));
    if (!verts) {
        fprintf(stderr, "  [warning] alloc failed in gpu time\n");
        p_glDeleteProgram(prog);
        return;
    }
    for (int i = 0; i < vert_count; i++) {
        verts[i * 3 + 0] = (float)((i % 7) - 3) * 0.1f;
        verts[i * 3 + 1] = (float)((i % 5) - 2) * 0.1f;
        verts[i * 3 + 2] = 0.0f;
    }

    GLuint vao, vbo;
    p_glGenVertexArrays(1, &vao);
    p_glBindVertexArray(vao);
    p_glGenBuffers(1, &vbo);
    p_glBindBuffer(GL_ARRAY_BUFFER, vbo);
    p_glBufferData(GL_ARRAY_BUFFER, (size_t)vert_count * 3 * sizeof(float),
                   verts, GL_STATIC_DRAW);
    p_glEnableVertexAttribArray(0);
    p_glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                            (void *)0);

    const int warmup = 10;
    const int N = 100;

    /* Warm up. */
    for (int i = 0; i < warmup; i++) {
        p_glDrawArrays(GL_TRIANGLES, 0, vert_count);
    }
    p_glFinish();

    /* Phase 1: Measure CPU-only submit time (no GPU wait).
     * The draw commands are queued into the Metal command buffer but we
     * don't wait for GPU completion — this is pure CPU-side MGL overhead. */
    uint64_t cpu_start = now_ns();
    for (int i = 0; i < N; i++) {
        p_glDrawArrays(GL_TRIANGLES, 0, vert_count);
    }
    uint64_t cpu_end = now_ns();

    /* Phase 2: Measure CPU+GPU time (with glFinish to force GPU sync).
     * glFinish() blocks until all queued GPU work completes, so the
     * wall-clock time includes both CPU submission and GPU execution. */
    p_glFinish(); /* drain any prior work first */
    uint64_t total_start = now_ns();
    for (int i = 0; i < N; i++) {
        p_glDrawArrays(GL_TRIANGLES, 0, vert_count);
    }
    p_glFinish();
    uint64_t total_end = now_ns();

    uint64_t cpu_ns = cpu_end - cpu_start;
    uint64_t total_ns = total_end - total_start;
    /* GPU time = total - cpu_submit. Clamp to 0 if negative (measurement
     * noise when GPU work is negligible). */
    uint64_t gpu_ns = (total_ns > cpu_ns) ? (total_ns - cpu_ns) : 0;

    double cpu_us_per_draw = (double)cpu_ns / (double)N / 1000.0;
    double gpu_us_per_draw = (double)gpu_ns / (double)N / 1000.0;
    double ratio = (cpu_ns > 0) ? (double)gpu_ns / (double)cpu_ns : 0.0;

    print_row("GPU Time", "CPU submit us/draw", "%.3f", cpu_us_per_draw);
    print_row("GPU Time", "GPU exec us/draw", "%.3f", gpu_us_per_draw);
    print_row("GPU Time", "GPU/CPU ratio", "%.3f", ratio);

    p_glDeleteBuffers(1, &vbo);
    p_glDeleteVertexArrays(1, &vao);
    p_glDeleteProgram(prog);
    free(verts);
}

/* ======================================================================== */
/*  Benchmark 11: API Dispatch                                               */
/*  Measures pure CPU-side dispatch overhead of common GL entry points       */
/*  that do NOT trigger GPU work.  Each call binds/queries the SAME state    */
/*  it already holds (no-op from MGL's perspective) so the cost measured     */
/*  is purely the dispatch + state-compare path in the translation layer.    */
/*  This isolates MGL overhead from GPU pipeline cost and is the most        */
/*  sensitive benchmark for tracking dispatch-table / state-cache regressions. */
/* ======================================================================== */

#define DISPATCH_N 200000

/* Helper: run a no-op dispatch loop N times and report ns/call. */
#define DISPATCH_LOOP(label, call_expr, n)                                    \
    do {                                                                      \
        for (int _i = 0; _i < 2000; _i++) { call_expr; }                     \
        uint64_t _start = now_ns();                                           \
        for (int _i = 0; _i < (n); _i++) { call_expr; }                      \
        uint64_t _elapsed = now_ns() - _start;                               \
        double _ns = (double)_elapsed / (double)(n);                         \
        print_row("Dispatch: " label, "ns/call", "%.1f", _ns);               \
    } while (0)

static void benchmark_api_dispatch(void)
{
    /* Set up resources so the no-op binds are valid (bind once before the
     * timing loop so the state is already current — the loop calls are
     * genuine no-ops). */
    GLuint prog = build_program(kVS_src, kFS_src);
    p_glUseProgram(prog);

    GLuint vao;
    p_glGenVertexArrays(1, &vao);
    p_glBindVertexArray(vao);

    GLuint vbo;
    p_glGenBuffers(1, &vbo);
    p_glBindBuffer(GL_ARRAY_BUFFER, vbo);

    GLuint tex;
    p_glGenTextures(1, &tex);
    p_glBindTexture(GL_TEXTURE_2D, tex);
    p_glActiveTexture(GL_TEXTURE0);

    /* --- Bind calls (no-op: same target already bound) --- */
    DISPATCH_LOOP("glUseProgram",       p_glUseProgram(prog),                 DISPATCH_N);
    DISPATCH_LOOP("glBindVertexArray",  p_glBindVertexArray(vao),             DISPATCH_N);
    DISPATCH_LOOP("glBindBuffer",       p_glBindBuffer(GL_ARRAY_BUFFER, vbo), DISPATCH_N);
    DISPATCH_LOOP("glBindTexture",      p_glBindTexture(GL_TEXTURE_2D, tex),  DISPATCH_N);
    DISPATCH_LOOP("glActiveTexture",    p_glActiveTexture(GL_TEXTURE0),       DISPATCH_N);

    /* --- State setters (no-op: same value already set) --- */
    DISPATCH_LOOP("glViewport",         p_glViewport(0, 0, 640, 480),         DISPATCH_N);
    DISPATCH_LOOP("glScissor",          p_glScissor(0, 0, 640, 480),          DISPATCH_N);
    DISPATCH_LOOP("glDepthMask",        p_glDepthMask(GL_TRUE),               DISPATCH_N);
    DISPATCH_LOOP("glColorMask",        p_glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE), DISPATCH_N);
    DISPATCH_LOOP("glCullFace",         p_glCullFace(GL_BACK),                DISPATCH_N);
    DISPATCH_LOOP("glPolygonMode",      p_glPolygonMode(GL_FRONT_AND_BACK, GL_FILL), DISPATCH_N);
    DISPATCH_LOOP("glPixelStorei",      p_glPixelStorei(GL_UNPACK_ALIGNMENT, 4), DISPATCH_N);

    /* --- Vertex attrib array toggle (enable+disable pair) --- */
    {
        const int n = DISPATCH_N;
        for (int i = 0; i < 2000; i++) {
            p_glEnableVertexAttribArray(0);
            p_glDisableVertexAttribArray(0);
        }
        uint64_t start = now_ns();
        for (int i = 0; i < n; i++) {
            p_glEnableVertexAttribArray(0);
            p_glDisableVertexAttribArray(0);
        }
        uint64_t elapsed = now_ns() - start;
        double ns_per_call = (double)elapsed / (double)(n * 2);
        print_row("Dispatch: AttribArray Toggle", "ns/call", "%.1f", ns_per_call);
    }

    /* --- Query path ( glGetIntegerv — pure CPU state read ) --- */
    {
        GLint val = 0;
        DISPATCH_LOOP("glGetIntegerv",   p_glGetIntegerv(GL_VIEWPORT, &val),  DISPATCH_N);
        (void)val;
    }

    /* --- GetError (common in debug builds — measure its cost) --- */
    DISPATCH_LOOP("glGetError",         p_glGetError(),                       DISPATCH_N);

    p_glDeleteTextures(1, &tex);
    p_glDeleteBuffers(1, &vbo);
    p_glDeleteVertexArrays(1, &vao);
    p_glDeleteProgram(prog);
}

#undef DISPATCH_N

/* ======================================================================== */
/*  Benchmark 12: Fine-grained GPU Time                                      */
/*  Decomposes a frame into CPU-submit vs GPU-exec cost using staged          */
/*  glFinish() barriers, then measures per-draw GPU marginal cost via         */
/*  draw-count scaling (1 draw vs N draws, differencing out fixed overhead).  */
/*  MGL's GL_TIMESTAMP query is a stub (returns fake counters), so we cannot  */
/*  use GPU timestamp queries; instead we use the differential glFinish       */
/*  technique which works with any GL implementation.                         */
/* ======================================================================== */

static void benchmark_gpu_time_fine(void)
{
    GLuint prog = build_program(kVS_src, kFS_src);
    p_glUseProgram(prog);
    GLint loc = p_glGetUniformLocation(prog, "uColor");
    p_glUniform4f(loc, 0.2f, 0.4f, 0.8f, 1.0f);

    /* Shared vertex data: a single triangle. */
    const float verts[9] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f,
    };
    GLuint vao, vbo;
    p_glGenVertexArrays(1, &vao);
    p_glBindVertexArray(vao);
    p_glGenBuffers(1, &vbo);
    p_glBindBuffer(GL_ARRAY_BUFFER, vbo);
    p_glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    p_glEnableVertexAttribArray(0);
    p_glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);

    const int warmup = 20;
    const int samples = 200;   /* averaged samples per measurement */
    const int draw_counts[] = {1, 10, 100, 500};
    const int num_dc = (int)(sizeof(draw_counts) / sizeof(draw_counts[0]));

    /* Warm up. */
    for (int i = 0; i < warmup; i++) {
        p_glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    p_glFinish();

    /* Phase 1: CPU submit time per draw (no finish — pure dispatch overhead).
     * Measured at draw_counts[0] (1 draw) so per-draw cost is direct. */
    {
        uint64_t total = 0;
        for (int s = 0; s < samples; s++) {
            uint64_t start = now_ns();
            for (int i = 0; i < draw_counts[0]; i++) {
                p_glDrawArrays(GL_TRIANGLES, 0, 3);
            }
            total += now_ns() - start;
        }
        double cpu_ns = (double)total / (double)samples / (double)draw_counts[0];
        print_row("GPU Fine: CPU submit/draw", "us", "%.3f", cpu_ns / 1000.0);
    }

    /* Phase 2: End-to-end time per draw (submit + glFinish) at each draw count.
     * The per-draw GPU marginal cost is derived from the slope between
     * successive draw counts:  gpu_marginal = (T(N) - T(1)) / (N - 1) - cpu_submit.
     * This cancels out the fixed glFinish / command-buffer commit overhead. */
    double e2e_per_draw[8];  /* max draw_counts entries */
    if (num_dc > 8) { /* should not happen */ }
    for (int d = 0; d < num_dc; d++) {
        int dc = draw_counts[d];
        uint64_t total = 0;
        for (int s = 0; s < samples; s++) {
            p_glFinish(); /* drain before each sample */
            uint64_t start = now_ns();
            for (int i = 0; i < dc; i++) {
                p_glDrawArrays(GL_TRIANGLES, 0, 3);
            }
            p_glFinish();
            total += now_ns() - start;
        }
        e2e_per_draw[d] = (double)total / (double)samples / (double)dc;
        char metric_label[64];
        snprintf(metric_label, sizeof(metric_label), "us (N=%d)", dc);
        print_row("GPU Fine: E2E/draw", metric_label, "%.3f", e2e_per_draw[d] / 1000.0);
    }

    /* Phase 3: GPU marginal cost from slope (N=1 vs N=500). */
    if (num_dc >= 2) {
        double t1 = e2e_per_draw[0];              /* per-draw at N=1 */
        double tN = e2e_per_draw[num_dc - 1];     /* per-draw at N=max */
        int N = draw_counts[num_dc - 1];
        /* total_e2e(N) = fixed + N * (cpu + gpu_marginal)
         * total_e2e(1) = fixed + 1 * (cpu + gpu_marginal)
         * => total_e2e(N) - total_e2e(1) = (N-1) * (cpu + gpu_marginal)
         * => cpu + gpu_marginal = (totalN - total1) / (N-1)
         * Note total = per_draw * count, so:
         *   totalN = tN * N, total1 = t1 * 1
         *   cpu + gpu_marginal = (tN * N - t1) / (N - 1)
         * But we measured cpu_submit separately (Phase 1 at N=1, no finish).
         * To keep the derivation clean, report the combined marginal. */
        double marginal_us = (tN * (double)N - t1) / (double)(N - 1) / 1000.0;
        print_row("GPU Fine: Marginal/draw", "us (slope)", "%.3f", marginal_us);

        /* GPU-only marginal = combined marginal - cpu_submit (from Phase 1).
         * Phase 1 cpu_ns is in ns; convert to us.  We re-derive cpu from the
         * N=1 no-finish measurement stored implicitly: recompute here. */
        {
            uint64_t cpu_total = 0;
            for (int s = 0; s < samples; s++) {
                uint64_t start = now_ns();
                for (int i = 0; i < N; i++) {
                    p_glDrawArrays(GL_TRIANGLES, 0, 3);
                }
                cpu_total += now_ns() - start;
            }
            double cpu_per_draw_us = (double)cpu_total / (double)samples / (double)N / 1000.0;
            double gpu_marginal_us = marginal_us - cpu_per_draw_us;
            if (gpu_marginal_us < 0.0) gpu_marginal_us = 0.0;
            char cpu_metric[64];
            snprintf(cpu_metric, sizeof(cpu_metric), "us (N=%d)", N);
            print_row("GPU Fine: CPU submit/draw", cpu_metric, "%.3f", cpu_per_draw_us);
            print_row("GPU Fine: GPU exec/draw", "us (marginal)", "%.3f", gpu_marginal_us);
        }
    }

    /* Phase 4: Command buffer commit overhead (fixed cost per flush).
     * Compare N draws + 1 finish  vs  N * (1 draw + 1 finish).
     * The difference is (N-1) extra command-buffer commits. */
    {
        int N = 100;
        uint64_t batch_total = 0, indiv_total = 0;

        for (int s = 0; s < samples; s++) {
            p_glFinish();
            uint64_t start = now_ns();
            for (int i = 0; i < N; i++) {
                p_glDrawArrays(GL_TRIANGLES, 0, 3);
            }
            p_glFinish();
            batch_total += now_ns() - start;
        }
        for (int s = 0; s < samples; s++) {
            p_glFinish();
            uint64_t start = now_ns();
            for (int i = 0; i < N; i++) {
                p_glDrawArrays(GL_TRIANGLES, 0, 3);
                p_glFinish();
            }
            indiv_total += now_ns() - start;
        }
        /* indiv - batch = (N-1) * commit_overhead
         * (the Nth finish is present in both) */
        double commit_ns = (double)(indiv_total - batch_total) / (double)samples / (double)(N - 1);
        if (commit_ns < 0.0) commit_ns = 0.0;
        print_row("GPU Fine: CB commit", "us", "%.3f", commit_ns / 1000.0);
    }

    p_glDeleteBuffers(1, &vbo);
    p_glDeleteVertexArrays(1, &vao);
    p_glDeleteProgram(prog);
}

/* ======================================================================== */
/*  Benchmark 13: End-to-End                                                 */
/*  Simulates a Minecraft-like frame for up to 60 s / 3600 frames.          */
/* ======================================================================== */

static void benchmark_end_to_end(void)
{
    /* Three shader programs: terrain, entity, gui.  Terrain and entity
     * reuse kVS2/kFS2 (have uMVP + uColor); gui uses kVS/kFS (uColor only).
     * The point is to measure the cost of switching programs, not visual
     * output, so reusing the same shader sources is fine. */
    GLuint prog_terrain = build_program(kVS2_src, kFS2_src);
    GLuint prog_entity  = build_program(kVS2_src, kFS2_src);
    GLuint prog_gui     = build_program(kVS_src,  kFS_src);

    GLint terrain_mvp   = p_glGetUniformLocation(prog_terrain, "uMVP");
    GLint terrain_color = p_glGetUniformLocation(prog_terrain, "uColor");
    GLint entity_mvp    = p_glGetUniformLocation(prog_entity,  "uMVP");
    GLint entity_color  = p_glGetUniformLocation(prog_entity,  "uColor");
    GLint gui_color     = p_glGetUniformLocation(prog_gui,     "uColor");

    /* Cube geometry: 36 vertices (6 faces, 2 triangles each). */
    const float cube_verts[] = {
        /* front */
        -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,
        -0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,
        /* back */
         0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f,  0.5f, -0.5f,
         0.5f, -0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f,
        /* left */
        -0.5f, -0.5f, -0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f,
        -0.5f, -0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f,
        /* right */
         0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,
         0.5f, -0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,
        /* bottom */
        -0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,
        -0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f,
        /* top */
        -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f,
        -0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f,
    };
    const int cube_vert_count = 36;

    /* Quad geometry: 6 vertices (2 triangles). */
    const float quad_verts[] = {
        -0.5f, -0.5f, 0.0f,  0.5f, -0.5f, 0.0f,  0.5f,  0.5f, 0.0f,
        -0.5f, -0.5f, 0.0f,  0.5f,  0.5f, 0.0f, -0.5f,  0.5f, 0.0f,
    };
    const int quad_vert_count = 6;

    /* Cube VAO. */
    GLuint vao, vbo;
    p_glGenVertexArrays(1, &vao);
    p_glBindVertexArray(vao);
    p_glGenBuffers(1, &vbo);
    p_glBindBuffer(GL_ARRAY_BUFFER, vbo);
    p_glBufferData(GL_ARRAY_BUFFER, sizeof(cube_verts), cube_verts,
                   GL_STATIC_DRAW);
    p_glEnableVertexAttribArray(0);
    p_glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                            (void *)0);

    /* Quad VAO. */
    GLuint vao_quad, vbo_quad;
    p_glGenVertexArrays(1, &vao_quad);
    p_glBindVertexArray(vao_quad);
    p_glGenBuffers(1, &vbo_quad);
    p_glBindBuffer(GL_ARRAY_BUFFER, vbo_quad);
    p_glBufferData(GL_ARRAY_BUFFER, sizeof(quad_verts), quad_verts,
                   GL_STATIC_DRAW);
    p_glEnableVertexAttribArray(0);
    p_glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                            (void *)0);

    /* 8 textures of 64x64 RGBA8, cycled through during the frame. */
    const int tex_size = 64;
    const size_t tex_bytes = (size_t)tex_size * tex_size * 4;
    uint8_t *tex_data = (uint8_t *)malloc(tex_bytes);
    if (!tex_data) {
        fprintf(stderr, "  [warning] alloc failed in end-to-end\n");
        p_glDeleteBuffers(1, &vbo_quad);
        p_glDeleteVertexArrays(1, &vao_quad);
        p_glDeleteBuffers(1, &vbo);
        p_glDeleteVertexArrays(1, &vao);
        p_glDeleteProgram(prog_terrain);
        p_glDeleteProgram(prog_entity);
        p_glDeleteProgram(prog_gui);
        return;
    }
    memset(tex_data, 0xFF, tex_bytes);

    const int num_tex = 8;
    GLuint textures[8];
    p_glGenTextures(num_tex, textures);
    for (int i = 0; i < num_tex; i++) {
        p_glActiveTexture(GL_TEXTURE0 + (GLenum)i);
        p_glBindTexture(GL_TEXTURE_2D, textures[i]);
        p_glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, tex_size, tex_size, 0,
                       GL_RGBA, GL_UNSIGNED_BYTE, tex_data);
        p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    p_glViewport(0, 0, 640, 480);
    p_glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    const int max_frames = 3600;
    const uint64_t max_duration_ns = 60ULL * 1000ULL * 1000ULL * 1000ULL;

    /* Warm up: exercise all three shader pipelines + both VAOs. */
    for (int f = 0; f < 3; f++) {
        float wmvp[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        p_glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        p_glEnable(GL_DEPTH_TEST);
        p_glUseProgram(prog_terrain);
        p_glBindVertexArray(vao);
        p_glUniformMatrix4fv(terrain_mvp, 1, GL_FALSE, wmvp);
        p_glUniform4f(terrain_color, 0.5f, 0.5f, 0.5f, 1.0f);
        p_glActiveTexture(GL_TEXTURE0);
        p_glBindTexture(GL_TEXTURE_2D, textures[0]);
        p_glDrawArrays(GL_TRIANGLES, 0, cube_vert_count);

        p_glUseProgram(prog_entity);
        p_glUniformMatrix4fv(entity_mvp, 1, GL_FALSE, wmvp);
        p_glUniform4f(entity_color, 0.8f, 0.2f, 0.2f, 1.0f);
        p_glDrawArrays(GL_TRIANGLES, 0, cube_vert_count);

        p_glDisable(GL_DEPTH_TEST);
        p_glEnable(GL_BLEND);
        p_glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        p_glUseProgram(prog_gui);
        p_glBindVertexArray(vao_quad);
        p_glUniform4f(gui_color, 0.9f, 0.9f, 0.9f, 0.8f);
        p_glDrawArrays(GL_TRIANGLES, 0, quad_vert_count);
        p_glDisable(GL_BLEND);

        glfwSwapBuffers(g_window);
    }

    double *frame_times = (double *)malloc((size_t)max_frames * sizeof(double));
    if (!frame_times) {
        fprintf(stderr, "  [warning] alloc failed in end-to-end timing\n");
        free(tex_data);
        p_glDeleteTextures(num_tex, textures);
        p_glDeleteBuffers(1, &vbo_quad);
        p_glDeleteVertexArrays(1, &vao_quad);
        p_glDeleteBuffers(1, &vbo);
        p_glDeleteVertexArrays(1, &vao);
        p_glDeleteProgram(prog_terrain);
        p_glDeleteProgram(prog_entity);
        p_glDeleteProgram(prog_gui);
        return;
    }

    int frames = 0;
    uint64_t loop_start = now_ns();

    while (frames < max_frames) {
        uint64_t frame_start = now_ns();

        p_glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        /* --- Terrain pass: 100 draws, depth test on. --- */
        p_glEnable(GL_DEPTH_TEST);
        p_glUseProgram(prog_terrain);
        p_glBindVertexArray(vao);
        for (int i = 0; i < 100; i++) {
            /* Different model matrix per chunk: Y-rotation that increases
             * per draw, plus a translation offset.  Stored column-major. */
            float angle = (float)(i + frames) * 0.05f;
            float c = cosf(angle), s = sinf(angle);
            float mvp[16] = {
                c, 0, -s, 0,
                0, 1,  0, 0,
                s, 0,  c, 0,
                (float)(i % 10) * 0.1f, 0,
                (float)(i / 10) * 0.1f, 1
            };
            p_glUniformMatrix4fv(terrain_mvp, 1, GL_FALSE, mvp);
            p_glUniform4f(terrain_color, 0.4f, 0.6f, 0.3f, 1.0f);

            /* Bind a different texture every few draws. */
            p_glActiveTexture(GL_TEXTURE0);
            p_glBindTexture(GL_TEXTURE_2D, textures[i % num_tex]);

            /* Toggle depth test enable/disable every ~10 draws. */
            if (i > 0 && (i % 10) == 0) {
                p_glDisable(GL_DEPTH_TEST);
                p_glEnable(GL_DEPTH_TEST);
            }
            /* Change blend func every ~20 draws. */
            if (i > 0 && (i % 20) == 0) {
                p_glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                p_glBlendFunc(GL_ONE, GL_ZERO);
            }

            p_glDrawArrays(GL_TRIANGLES, 0, cube_vert_count);
        }

        /* --- Entity pass: 30 draws, depth test on. --- */
        p_glUseProgram(prog_entity);
        p_glBindVertexArray(vao);
        for (int i = 0; i < 30; i++) {
            float angle = (float)(i + frames) * 0.1f;
            float c = cosf(angle), s = sinf(angle);
            float mvp[16] = {
                c, 0, -s, 0,
                0, 1,  0, 0,
                s, 0,  c, 0,
                (float)(i % 5) * 0.2f,
                (float)(i % 3) * 0.1f, 0, 1
            };
            p_glUniformMatrix4fv(entity_mvp, 1, GL_FALSE, mvp);
            p_glUniform4f(entity_color, 0.8f, 0.2f, 0.2f, 1.0f);

            p_glActiveTexture(GL_TEXTURE0);
            p_glBindTexture(GL_TEXTURE_2D, textures[i % num_tex]);

            p_glDrawArrays(GL_TRIANGLES, 0, cube_vert_count);
        }

        /* --- GUI pass: 10 draws, depth test off, blend on. --- */
        p_glDisable(GL_DEPTH_TEST);
        p_glEnable(GL_BLEND);
        p_glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        p_glUseProgram(prog_gui);
        p_glBindVertexArray(vao_quad);
        for (int i = 0; i < 10; i++) {
            p_glUniform4f(gui_color, 0.9f, 0.9f, 0.9f, 0.8f);
            p_glActiveTexture(GL_TEXTURE0);
            p_glBindTexture(GL_TEXTURE_2D, textures[i % num_tex]);
            p_glDrawArrays(GL_TRIANGLES, 0, quad_vert_count);
        }
        p_glDisable(GL_BLEND);
        p_glEnable(GL_DEPTH_TEST);

        /* --- Animated texture upload every ~30 frames. --- */
        if ((frames % 30) == 0) {
            memset(tex_data, (uint8_t)(frames & 0xFF), tex_bytes);
            p_glActiveTexture(GL_TEXTURE0);
            p_glBindTexture(GL_TEXTURE_2D, textures[0]);
            p_glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tex_size, tex_size,
                              GL_RGBA, GL_UNSIGNED_BYTE, tex_data);
        }

        glfwSwapBuffers(g_window);

        uint64_t frame_end = now_ns();
        frame_times[frames] = (double)(frame_end - frame_start) / 1e6; /* ms */
        frames++;

        if ((frame_end - loop_start) >= max_duration_ns) {
            break;
        }
    }

    double total_ms = 0.0;
    double min_ms = frame_times[0];
    double max_ms = frame_times[0];
    for (int i = 0; i < frames; i++) {
        total_ms += frame_times[i];
        if (frame_times[i] < min_ms) min_ms = frame_times[i];
        if (frame_times[i] > max_ms) max_ms = frame_times[i];
    }
    double avg_ms = total_ms / (double)frames;
    double fps = 1000.0 / avg_ms;

    print_row("End-to-End", "FPS", "%.1f", fps);
    print_row("End-to-End", "Avg Frame ms", "%.2f", avg_ms);
    print_row("End-to-End", "Min Frame ms", "%.2f", min_ms);
    print_row("End-to-End", "Max Frame ms", "%.2f", max_ms);

    free(frame_times);
    free(tex_data);
    p_glDeleteTextures(num_tex, textures);
    p_glDeleteBuffers(1, &vbo_quad);
    p_glDeleteVertexArrays(1, &vao_quad);
    p_glDeleteBuffers(1, &vbo);
    p_glDeleteVertexArrays(1, &vao);
    p_glDeleteProgram(prog_terrain);
    p_glDeleteProgram(prog_entity);
    p_glDeleteProgram(prog_gui);
}

/* ======================================================================== */
/*  Main                                                                     */
/* ======================================================================== */

static void print_usage(const char *prog)
{
    printf("Usage: %s [--help]\n", prog);
    printf("\n");
    printf("Runs the MGL benchmark suite (12 categories) and prints a\n");
    printf("results table.  The program creates a hidden 640x480 GLFW\n");
    printf("window, loads GL entry points through glfwGetProcAddress\n");
    printf("(routed through MGL), and measures translation overhead.\n");
    printf("\n");
    printf("Categories:\n");
    printf("  1.  Empty Draw         7.  Pipeline Switch\n");
    printf("  2.  Triangle Draw      8.  Uniform Update\n");
    printf("  3.  Batch Draw         9.  GPU Time (glFinish)\n");
    printf("  4.  Texture Upload    10.  End-to-End (Minecraft-like)\n");
    printf("  5.  Buffer Upload     11.  API Dispatch (no-op overhead)\n");
    printf("  6.  State Changes     12.  Fine-grained GPU Time\n");
}

int main(int argc, char **argv)
{
    if (argc > 1) {
        if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        /* Unknown argument — warn but continue. */
        fprintf(stderr, "Unknown argument: %s\n\n", argv[1]);
        print_usage(argv[0]);
        return 1;
    }

    init_timing();

    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    g_window = glfwCreateWindow(640, 480, "mgl_benchmark", NULL, NULL);
    if (!g_window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(g_window);
    glfwSwapInterval(0);

    if (!load_gl()) {
        fprintf(stderr, "Failed to load GL functions\n");
        glfwTerminate();
        return 1;
    }

    /* Print some context info for sanity. */
    const GLubyte *renderer = p_glGetString(GL_RENDERER);
    const GLubyte *version = p_glGetString(GL_VERSION);
    const GLubyte *vendor = p_glGetString(GL_VENDOR);
    printf("Renderer: %s\n", renderer ? (const char *)renderer : "(null)");
    printf("Version:  %s\n", version ? (const char *)version : "(null)");
    printf("Vendor:   %s\n", vendor ? (const char *)vendor : "(null)");
    printf("\n");

    p_glViewport(0, 0, 640, 480);
    check_gl_error("initial setup");

    print_table_header();

    benchmark_empty_draw();
    benchmark_triangle_draw();
    benchmark_batch_draw();
    benchmark_texture_upload();
    benchmark_buffer_upload();
    benchmark_state_changes();
    benchmark_pipeline_switch();
    benchmark_uniform_update();
    benchmark_gpu_time();
    benchmark_api_dispatch();
    benchmark_gpu_time_fine();
    benchmark_end_to_end();

    print_table_footer();

    glfwDestroyWindow(g_window);
    glfwTerminate();
    return 0;
}
