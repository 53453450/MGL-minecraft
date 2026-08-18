/*
 * mgl_benchmark.c
 *
 * Comprehensive benchmark program for measuring GL translation overhead.
 *
 * Build (MGL):
 *   make bench
 * Build (system Apple OpenGL):
 *   make bench-system
 *
 * The same source compiles for both backends.  When linked against MGL,
 * glfwGetProcAddress returns MGL's translation-layer entry points.
 * When linked against the system OpenGL framework (no -lmgl), the calls
 * go directly to Apple's native OpenGL driver.
 *
 * Timer Query support (GL_TIME_ELAPSED / GL_TIMESTAMP) is used for
 * accurate GPU execution time measurement.  MGL implements this via
 * Metal's sampleTimestamps:gpuTimestamp: API; the system OpenGL driver
 * provides its own native implementation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <math.h>
#include <errno.h>
#include <sys/utsname.h>

#include <mach/mach_time.h>

/* Prevent GLFW from pulling in the system OpenGL headers — we want MGL's
 * glcorearb.h to be the sole source of GL type/enum definitions so that the
 * function pointers we load via glfwGetProcAddress match MGL's signatures. */
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

/* glcorearb.h provides the GLenum/GLuint typedefs, the GL_* enum constants,
 * and the PFNGL* function-pointer typedefs. */
#define GL_GLEXT_PROTOTYPES 1
#include <GL/glcorearb.h>

/* ======================================================================== */
/*  Backend identification                                                   */
/* ======================================================================== */

#if defined(__MGL_BENCHMARK_SYSTEM_GL__)
#define BENCHMARK_BACKEND_NAME "System Apple OpenGL"
#else
#define BENCHMARK_BACKEND_NAME "MGL (Metal Translation Layer)"
#endif

/* ======================================================================== */
/*  GL function-pointer declarations                                         */
/* ======================================================================== */

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
static PFNGLBINDVERTEXBUFFERPROC     p_glBindVertexBuffer    = NULL;
static PFNGLVERTEXATTRIBFORMATPROC   p_glVertexAttribFormat  = NULL;
static PFNGLVERTEXATTRIBBINDINGPROC  p_glVertexAttribBinding = NULL;

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
static PFNGLUNIFORM1IPROC            p_glUniform1i           = NULL;
static PFNGLUNIFORM4FPROC            p_glUniform4f           = NULL;
static PFNGLUNIFORMMATRIX4FVPROC     p_glUniformMatrix4fv    = NULL;
static PFNGLGETUNIFORMBLOCKINDEXPROC p_glGetUniformBlockIndex = NULL;
static PFNGLUNIFORMBLOCKBINDINGPROC  p_glUniformBlockBinding = NULL;
static PFNGLBINDBUFFERRANGEPROC      p_glBindBufferRange     = NULL;

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
static PFNGLGETQUERYOBJECTIVPROC     p_glGetQueryObjectiv    = NULL;

/* API dispatch benchmark — additional no-op bind / query entry points. */
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

#ifndef MGL_BENCHMARK_GIT_COMMIT
#define MGL_BENCHMARK_GIT_COMMIT "unknown"
#endif

static const char *g_selected_test = NULL;
static const char *g_json_path = NULL;
static int g_frame_limit = 0;
static int g_warmup_frames = 10;
static const char *g_renderer_name = "unknown";
static const char *g_version_name = "unknown";
static const char *g_vendor_name = "unknown";

static void init_timing(void)
{
    mach_timebase_info(&g_timebase);
}

static uint64_t now_ns(void)
{
    uint64_t t = mach_absolute_time();
    return (uint64_t)((double)t * (double)g_timebase.numer /
                      (double)g_timebase.denom);
}

/* ======================================================================== */
/*  Result buffering — all results stored and printed at the end             */
/* ======================================================================== */

#define MAX_RESULTS 512
#define METIC_LEN   32
#define VALUE_LEN   32

typedef struct {
    char test[METIC_LEN];
    char metric[METIC_LEN];
    char value[VALUE_LEN];
} BenchmarkResult;

static BenchmarkResult g_results[MAX_RESULTS];
static int g_result_count = 0;

static void record_result(const char *test, const char *metric,
                          const char *fmt, ...)
{
    if (g_result_count >= MAX_RESULTS) {
        fprintf(stderr, "  [warning] result buffer full, dropping result\n");
        return;
    }
    BenchmarkResult *r = &g_results[g_result_count++];
    strncpy(r->test, test, METIC_LEN - 1);
    r->test[METIC_LEN - 1] = '\0';
    strncpy(r->metric, metric, METIC_LEN - 1);
    r->metric[METIC_LEN - 1] = '\0';
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(r->value, VALUE_LEN, fmt, ap);
    va_end(ap);
}

/* Print all buffered results in a formatted table. */
static void print_all_results(void)
{
    printf("\n");
    printf("========================================");
    printf("========================================\n");
    printf("  Benchmark Results — %s\n", BENCHMARK_BACKEND_NAME);
    printf("========================================");
    printf("========================================\n");
    printf("  %-28s %-24s %s\n", "Test", "Metric", "Value");
    printf("  %-28s %-24s %s\n",
           "----------------------------", "------------------------", "--------");
    for (int i = 0; i < g_result_count; i++) {
        printf("  %-28s %-24s %s\n",
               g_results[i].test,
               g_results[i].metric,
               g_results[i].value);
    }
    printf("========================================");
    printf("========================================\n");
}

static int compare_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static double percentile_sorted(const double *values, int count, double p)
{
    if (count <= 0) return 0.0;
    double position = p * (double)(count - 1);
    int lower = (int)floor(position);
    int upper = (int)ceil(position);
    double fraction = position - (double)lower;
    return values[lower] + (values[upper] - values[lower]) * fraction;
}

static void record_frame_statistics(const char *test, double *frame_times,
                                    int frames, int draws_per_frame)
{
    if (frames <= 0) return;

    double total_ms = 0.0;
    double min_ms = frame_times[0];
    double max_ms = frame_times[0];
    for (int i = 0; i < frames; i++) {
        total_ms += frame_times[i];
        if (frame_times[i] < min_ms) min_ms = frame_times[i];
        if (frame_times[i] > max_ms) max_ms = frame_times[i];
    }

    qsort(frame_times, (size_t)frames, sizeof(double), compare_double);
    double avg_ms = total_ms / (double)frames;
    record_result(test, "FPS", "%.1f", 1000.0 / avg_ms);
    record_result(test, "Avg Frame ms", "%.3f", avg_ms);
    record_result(test, "P50 Frame ms", "%.3f",
                  percentile_sorted(frame_times, frames, 0.50));
    record_result(test, "P95 Frame ms", "%.3f",
                  percentile_sorted(frame_times, frames, 0.95));
    record_result(test, "P99 Frame ms", "%.3f",
                  percentile_sorted(frame_times, frames, 0.99));
    record_result(test, "Min Frame ms", "%.3f", min_ms);
    record_result(test, "Max Frame ms", "%.3f", max_ms);
    if (draws_per_frame > 0) {
        record_result(test, "Draws/frame", "%d", draws_per_frame);
        record_result(test, "Draw/s", "%.0f",
                      (double)draws_per_frame * 1000.0 / avg_ms);
    }
}

static void record_duration_statistics(const char *test, const char *label,
                                       double *times, int count)
{
    if (count <= 0) return;
    double total = 0.0;
    for (int i = 0; i < count; i++) total += times[i];
    qsort(times, (size_t)count, sizeof(double), compare_double);

    char metric[METIC_LEN];
    snprintf(metric, sizeof(metric), "%s Avg ms", label);
    record_result(test, metric, "%.3f", total / (double)count);
    snprintf(metric, sizeof(metric), "%s P50 ms", label);
    record_result(test, metric, "%.3f", percentile_sorted(times, count, 0.50));
    snprintf(metric, sizeof(metric), "%s P95 ms", label);
    record_result(test, metric, "%.3f", percentile_sorted(times, count, 0.95));
    snprintf(metric, sizeof(metric), "%s P99 ms", label);
    record_result(test, metric, "%.3f", percentile_sorted(times, count, 0.99));
}

static void json_write_string(FILE *out, const char *value)
{
    fputc('"', out);
    for (const unsigned char *p = (const unsigned char *)(value ? value : "");
         *p; p++) {
        switch (*p) {
            case '"': fputs("\\\"", out); break;
            case '\\': fputs("\\\\", out); break;
            case '\n': fputs("\\n", out); break;
            case '\r': fputs("\\r", out); break;
            case '\t': fputs("\\t", out); break;
            default:
                if (*p < 0x20) fprintf(out, "\\u%04x", (unsigned)*p);
                else fputc(*p, out);
                break;
        }
    }
    fputc('"', out);
}

static int write_json_results(const char *path)
{
    if (!path) return 1;
    FILE *out = fopen(path, "w");
    if (!out) {
        fprintf(stderr, "Failed to open JSON output '%s': %s\n",
                path, strerror(errno));
        return 0;
    }

    struct utsname os;
    memset(&os, 0, sizeof(os));
    uname(&os);
    fputs("{\n  \"schema_version\": 1,\n  \"backend\": ", out);
    json_write_string(out, BENCHMARK_BACKEND_NAME);
    fputs(",\n  \"renderer\": ", out); json_write_string(out, g_renderer_name);
    fputs(",\n  \"gl_version\": ", out); json_write_string(out, g_version_name);
    fputs(",\n  \"vendor\": ", out); json_write_string(out, g_vendor_name);
    fputs(",\n  \"git_commit\": ", out); json_write_string(out, MGL_BENCHMARK_GIT_COMMIT);
    fputs(",\n  \"os\": ", out); json_write_string(out, os.release);
    fputs(",\n  \"machine\": ", out); json_write_string(out, os.machine);
    fputs(",\n  \"results\": [\n", out);
    for (int i = 0; i < g_result_count; i++) {
        char *end = NULL;
        double numeric = strtod(g_results[i].value, &end);
        fputs("    {\"test\": ", out); json_write_string(out, g_results[i].test);
        fputs(", \"metric\": ", out); json_write_string(out, g_results[i].metric);
        fprintf(out, ", \"value\": %.17g, \"display\": ", numeric);
        json_write_string(out, g_results[i].value);
        fprintf(out, "}%s\n", i + 1 == g_result_count ? "" : ",");
    }
    fputs("  ]\n}\n", out);
    if (fclose(out) != 0) {
        fprintf(stderr, "Failed to finalize JSON output '%s'\n", path);
        return 0;
    }
    return 1;
}

static int should_run(const char *name)
{
    return g_selected_test == NULL || strcmp(g_selected_test, name) == 0;
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
    /* Optional on the system OpenGL 4.1 comparison backend, required for the
     * Minecraft/Sodium per-draw binding path on MGL. */
    p_glBindVertexBuffer = (PFNGLBINDVERTEXBUFFERPROC)
        glfwGetProcAddress("glBindVertexBuffer");
    p_glVertexAttribFormat = (PFNGLVERTEXATTRIBFORMATPROC)
        glfwGetProcAddress("glVertexAttribFormat");
    p_glVertexAttribBinding = (PFNGLVERTEXATTRIBBINDINGPROC)
        glfwGetProcAddress("glVertexAttribBinding");

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
    LOAD_GL(p_glUniform1i,           PFNGLUNIFORM1IPROC,            "glUniform1i");
    LOAD_GL(p_glUniform4f,           PFNGLUNIFORM4FPROC,            "glUniform4f");
    LOAD_GL(p_glUniformMatrix4fv,    PFNGLUNIFORMMATRIX4FVPROC,     "glUniformMatrix4fv");
    LOAD_GL(p_glGetUniformBlockIndex, PFNGLGETUNIFORMBLOCKINDEXPROC, "glGetUniformBlockIndex");
    LOAD_GL(p_glUniformBlockBinding, PFNGLUNIFORMBLOCKBINDINGPROC,  "glUniformBlockBinding");
    LOAD_GL(p_glBindBufferRange,     PFNGLBINDBUFFERRANGEPROC,      "glBindBufferRange");

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
    LOAD_GL(p_glGetQueryObjectiv,    PFNGLGETQUERYOBJECTIVPROC,     "glGetQueryObjectiv");

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
/*  Timer Query helper — measures GPU elapsed time for bracketed commands    */
/* ======================================================================== */

/* Measures GPU nanoseconds elapsed between begin and end of a
 * GL_TIME_ELAPSED query.  Blocks until the result is available. */
static uint64_t measure_gpu_time_elapsed(void (*draw_fn)(void *ctx),
                                         void *ctx)
{
    GLuint q;
    p_glGenQueries(1, &q);

    p_glBeginQuery(GL_TIME_ELAPSED, q);
    draw_fn(ctx);
    p_glEndQuery(GL_TIME_ELAPSED);

    GLuint64 gpu_ns = 0;
    /* GL_QUERY_RESULT provides the required blocking behavior without a
     * userspace busy loop on asynchronous drivers. */
    p_glGetQueryObjectui64v(q, GL_QUERY_RESULT, &gpu_ns);
    p_glDeleteQueries(1, &q);
    return gpu_ns;
}

/* ======================================================================== */
/*  Global window handle (needed by swap-based benchmarks)                   */
/* ======================================================================== */

static GLFWwindow *g_window = NULL;

/* ======================================================================== */
/*  Benchmark 1: Empty Draw                                                  */
/* ======================================================================== */

static void benchmark_empty_draw(void)
{
    GLuint vao;
    p_glGenVertexArrays(1, &vao);
    p_glBindVertexArray(vao);

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
    record_result("Empty Draw", "ns/Draw", "%.1f", ns_per_draw);

    p_glBindVertexArray(0);
    p_glDeleteVertexArrays(1, &vao);
}

/* ======================================================================== */
/*  Benchmark 2: Triangle Draw                                               */
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

    record_result("Triangle Draw", "Draw/s", "%.0f", draws_per_s);
    record_result("Triangle Draw", "CPU us/draw", "%.2f", us_per_draw);

    p_glDeleteBuffers(1, &vbo);
    p_glDeleteVertexArrays(1, &vao);
    p_glDeleteProgram(prog);
}

/* ======================================================================== */
/*  Benchmark 3: Batch Draw (deferred vs immediate)                          */
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

    /* --- Batched (deferred) --- */
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

    /* --- Immediate: flush after every draw --- */
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

    record_result("Batch Draw (deferred)", "Draw/s", "%.0f", batched_dps);
    record_result("Batch Draw (immediate)", "Draw/s", "%.0f", immediate_dps);

    p_glDeleteBuffers(1, &vbo);
    p_glDeleteVertexArrays(1, &vao);
    p_glDeleteProgram(prog);
}

/* ======================================================================== */
/*  Benchmark 4: Texture Upload                                              */
/* ======================================================================== */

static void benchmark_texture_upload(void)
{
    const int W = 1024, H = 1024;
    const size_t data_size = (size_t)W * H * 4;

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

    record_result("Texture Upload", "MB/s", "%.1f", mb_per_s);
    record_result("Texture Upload", "us/call", "%.1f", us_per_call);

    free(pixels);
    p_glDeleteTextures(1, &tex);
}

/* ======================================================================== */
/*  Benchmark 5: Buffer Upload                                               */
/* ======================================================================== */

static void benchmark_buffer_upload(void)
{
    const size_t data_size = 1024 * 1024;

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

    record_result("Buffer Upload", "MB/s", "%.1f", mb_per_s);

    free(data);
    p_glDeleteBuffers(1, &buf);
}

/* ======================================================================== */
/*  Benchmark 6: State Changes                                               */
/* ======================================================================== */

static void benchmark_state_changes(void)
{
    const int N = 10000;

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
    double ns_per_call = (double)elapsed / (double)(N * 2);
    record_result("State: Depth Toggle", "ns/call", "%.1f", ns_per_call);

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
    record_result("State: BlendFunc Toggle", "ns/call", "%.1f", ns_per_call);
}

/* ======================================================================== */
/*  Benchmark 7: Pipeline Switch                                             */
/* ======================================================================== */

static void benchmark_pipeline_switch(void)
{
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
    if (p_glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "  [warning] pipeline-switch FBO A is incomplete\n");
    }

    p_glGenFramebuffers(1, &fboB);
    p_glBindFramebuffer(GL_FRAMEBUFFER, fboB);
    p_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, texB, 0);
    if (p_glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "  [warning] pipeline-switch FBO B is incomplete\n");
    }

    for (int i = 0; i < 100; i++) {
        p_glUseProgram(progA);
        p_glBindVertexArray(vaoA);
        p_glBindFramebuffer(GL_FRAMEBUFFER, fboA);
        p_glDrawArrays(GL_TRIANGLES, 0, 3);
        p_glUseProgram(progB);
        p_glBindVertexArray(vaoB);
        p_glBindFramebuffer(GL_FRAMEBUFFER, fboB);
        p_glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    p_glFlush();

    const int N = 1000;
    uint64_t start = now_ns();
    for (int i = 0; i < N; i++) {
        p_glUseProgram(progA);
        p_glBindVertexArray(vaoA);
        p_glBindFramebuffer(GL_FRAMEBUFFER, fboA);
        p_glDrawArrays(GL_TRIANGLES, 0, 3);

        p_glUseProgram(progB);
        p_glBindVertexArray(vaoB);
        p_glBindFramebuffer(GL_FRAMEBUFFER, fboB);
        p_glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    p_glFlush();
    uint64_t elapsed = now_ns() - start;

    double us_per_switch = (double)elapsed / 1000.0 / (double)(N * 2);
    record_result("Pipeline Switch + Draw", "CPU us/switch", "%.2f", us_per_switch);

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
/* ======================================================================== */

static void benchmark_uniform_update(void)
{
    GLuint prog = build_program(kVS2_src, kFS2_src);
    p_glUseProgram(prog);

    GLint loc_vec4 = p_glGetUniformLocation(prog, "uColor");
    GLint loc_mat4 = p_glGetUniformLocation(prog, "uMVP");

    float matrix[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    const int N = 10000;

    for (int i = 0; i < 1000; i++) {
        p_glUniform4f(loc_vec4, 1.0f, 0.5f, 0.25f, 1.0f);
    }

    uint64_t start = now_ns();
    for (int i = 0; i < N; i++) {
        p_glUniform4f(loc_vec4, 1.0f, 0.5f, 0.25f, 1.0f);
    }
    uint64_t elapsed = now_ns() - start;
    double ns_per_call = (double)elapsed / (double)N;
    record_result("Uniform: glUniform4f", "ns/call", "%.1f", ns_per_call);

    for (int i = 0; i < 1000; i++) {
        p_glUniformMatrix4fv(loc_mat4, 1, GL_FALSE, matrix);
    }

    start = now_ns();
    for (int i = 0; i < N; i++) {
        p_glUniformMatrix4fv(loc_mat4, 1, GL_FALSE, matrix);
    }
    elapsed = now_ns() - start;
    ns_per_call = (double)elapsed / (double)N;
    record_result("Uniform: glUniformMatrix4fv", "ns/call", "%.1f", ns_per_call);

    p_glDeleteProgram(prog);
}

/* ======================================================================== */
/*  Benchmark 9: GPU Time (Timer Query)                                      */
/*  Uses GL_TIME_ELAPSED query to measure actual GPU execution time.         */
/* ======================================================================== */

/* Context for the draw callback used by measure_gpu_time_elapsed. */
typedef struct {
    GLuint vao;
    GLsizei vert_count;
} DrawCtx;

static void draw_triangles_ctx(void *userdata)
{
    DrawCtx *d = (DrawCtx *)userdata;
    p_glDrawArrays(GL_TRIANGLES, 0, d->vert_count);
}

static void benchmark_gpu_time(void)
{
    GLuint prog = build_program(kVS_src, kFS_src);
    p_glUseProgram(prog);
    GLint loc = p_glGetUniformLocation(prog, "uColor");
    p_glUniform4f(loc, 0.2f, 0.4f, 0.8f, 1.0f);

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

    DrawCtx dctx = { vao, vert_count };

    /* Warm up. */
    for (int i = 0; i < 10; i++) {
        p_glDrawArrays(GL_TRIANGLES, 0, vert_count);
    }
    p_glFinish();

    /* --- CPU submit time (no GPU wait) --- */
    const int N = 100;
    uint64_t cpu_start = now_ns();
    for (int i = 0; i < N; i++) {
        p_glDrawArrays(GL_TRIANGLES, 0, vert_count);
    }
    uint64_t cpu_end = now_ns();
    double cpu_us_per_draw = (double)(cpu_end - cpu_start) / (double)N / 1000.0;

    /* --- GPU execution time via Timer Query ---
     * Measure GPU time for a single draw using GL_TIME_ELAPSED.
     * Average over multiple samples for stability. */
    const int samples = 30;
    uint64_t gpu_total_ns = 0;
    for (int i = 0; i < samples; i++) {
        gpu_total_ns += measure_gpu_time_elapsed(draw_triangles_ctx, &dctx);
    }
    double gpu_us_per_draw = (double)gpu_total_ns / (double)samples / 1000.0;

    /* --- GPU time for N draws via Timer Query ---
     * Measure total GPU time for N draws to get per-draw marginal cost. */
    GLuint q_batch;
    p_glGenQueries(1, &q_batch);

    /* Warm up. */
    p_glDrawArrays(GL_TRIANGLES, 0, vert_count);
    p_glFinish();

    p_glBeginQuery(GL_TIME_ELAPSED, q_batch);
    for (int i = 0; i < N; i++) {
        p_glDrawArrays(GL_TRIANGLES, 0, vert_count);
    }
    p_glEndQuery(GL_TIME_ELAPSED);

    GLuint64 batch_gpu_ns = 0;
    p_glGetQueryObjectui64v(q_batch, GL_QUERY_RESULT, &batch_gpu_ns);
    p_glDeleteQueries(1, &q_batch);

    double batch_gpu_us_per_draw = (double)batch_gpu_ns / (double)N / 1000.0;
    double ratio = (cpu_us_per_draw > 0.0)
        ? (gpu_us_per_draw / cpu_us_per_draw) : 0.0;

    record_result("GPU Time", "CPU submit us/draw", "%.3f", cpu_us_per_draw);
    record_result("GPU Time", "Query elapsed us/draw (1)", "%.3f", gpu_us_per_draw);
    record_result("GPU Time", "Query elapsed us/draw (N)", "%.3f", batch_gpu_us_per_draw);
    record_result("GPU Time", "Query/CPU ratio", "%.3f", ratio);

    p_glDeleteBuffers(1, &vbo);
    p_glDeleteVertexArrays(1, &vao);
    p_glDeleteProgram(prog);
    free(verts);
}

/* ======================================================================== */
/*  Benchmark 10: API Dispatch                                               */
/* ======================================================================== */

#define DISPATCH_N 200000

#define DISPATCH_LOOP(label, call_expr, n)                                    \
    do {                                                                      \
        for (int _i = 0; _i < 2000; _i++) { call_expr; }                     \
        uint64_t _start = now_ns();                                           \
        for (int _i = 0; _i < (n); _i++) { call_expr; }                      \
        uint64_t _elapsed = now_ns() - _start;                               \
        double _ns = (double)_elapsed / (double)(n);                         \
        record_result("Dispatch: " label, "ns/call", "%.1f", _ns);           \
    } while (0)

static void benchmark_api_dispatch(void)
{
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

    DISPATCH_LOOP("glUseProgram",       p_glUseProgram(prog),                 DISPATCH_N);
    DISPATCH_LOOP("glBindVertexArray",  p_glBindVertexArray(vao),             DISPATCH_N);
    DISPATCH_LOOP("glBindBuffer",       p_glBindBuffer(GL_ARRAY_BUFFER, vbo), DISPATCH_N);
    DISPATCH_LOOP("glBindTexture",      p_glBindTexture(GL_TEXTURE_2D, tex),  DISPATCH_N);
    DISPATCH_LOOP("glActiveTexture",    p_glActiveTexture(GL_TEXTURE0),       DISPATCH_N);

    DISPATCH_LOOP("glViewport",         p_glViewport(0, 0, 640, 480),         DISPATCH_N);
    DISPATCH_LOOP("glScissor",          p_glScissor(0, 0, 640, 480),          DISPATCH_N);
    DISPATCH_LOOP("glDepthMask",        p_glDepthMask(GL_TRUE),               DISPATCH_N);
    DISPATCH_LOOP("glColorMask",        p_glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE), DISPATCH_N);
    DISPATCH_LOOP("glCullFace",         p_glCullFace(GL_BACK),                DISPATCH_N);
    DISPATCH_LOOP("glPolygonMode",      p_glPolygonMode(GL_FRONT_AND_BACK, GL_FILL), DISPATCH_N);
    DISPATCH_LOOP("glPixelStorei",      p_glPixelStorei(GL_UNPACK_ALIGNMENT, 4), DISPATCH_N);

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
        record_result("Dispatch: AttribArray Toggle", "ns/call", "%.1f", ns_per_call);
    }

    {
        GLint viewport[4] = {0, 0, 0, 0};
        DISPATCH_LOOP("glGetIntegerv", p_glGetIntegerv(GL_VIEWPORT, viewport), DISPATCH_N);
        (void)viewport;
    }

    DISPATCH_LOOP("glGetError",         p_glGetError(),                       DISPATCH_N);

    p_glDeleteTextures(1, &tex);
    p_glDeleteBuffers(1, &vbo);
    p_glDeleteVertexArrays(1, &vao);
    p_glDeleteProgram(prog);
}

#undef DISPATCH_N

/* ======================================================================== */
/*  Benchmark 11: Fine-grained GPU Time (Timer Query)                        */
/*  Uses GL_TIME_ELAPSED to measure per-draw GPU marginal cost via           */
/*  draw-count scaling.                                                      */
/* ======================================================================== */

static void benchmark_gpu_time_fine(void)
{
    GLuint prog = build_program(kVS_src, kFS_src);
    p_glUseProgram(prog);
    GLint loc = p_glGetUniformLocation(prog, "uColor");
    p_glUniform4f(loc, 0.2f, 0.4f, 0.8f, 1.0f);

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
    const int samples = 30;
    const int draw_counts[] = {1, 10, 100, 500};
    const int num_dc = (int)(sizeof(draw_counts) / sizeof(draw_counts[0]));

    /* Warm up. */
    for (int i = 0; i < warmup; i++) {
        p_glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    p_glFinish();

    /* Phase 1: CPU submit time per draw (no GPU wait). */
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
        record_result("GPU Fine: CPU submit/draw", "us", "%.3f", cpu_ns / 1000.0);
    }

    /* Phase 2: GPU time per draw at each draw count via Timer Query.
     * For each draw count, measure the total GPU time for N draws using
     * GL_TIME_ELAPSED, then divide by N to get per-draw GPU cost. */
    double e2e_per_draw[8];
    if (num_dc > 8) { /* should not happen */ }
    for (int d = 0; d < num_dc; d++) {
        int dc = draw_counts[d];
        uint64_t gpu_total_ns = 0;
        for (int s = 0; s < samples; s++) {
            GLuint q;
            p_glGenQueries(1, &q);
            p_glBeginQuery(GL_TIME_ELAPSED, q);
            for (int i = 0; i < dc; i++) {
                p_glDrawArrays(GL_TRIANGLES, 0, 3);
            }
            p_glEndQuery(GL_TIME_ELAPSED);

            GLuint64 result = 0;
            p_glGetQueryObjectui64v(q, GL_QUERY_RESULT, &result);
            p_glDeleteQueries(1, &q);
            gpu_total_ns += result;
        }
        e2e_per_draw[d] = (double)gpu_total_ns / (double)samples / (double)dc;
        char metric_label[64];
        snprintf(metric_label, sizeof(metric_label), "us (N=%d)", dc);
        record_result("GPU Fine: GPU/draw", metric_label, "%.3f", e2e_per_draw[d] / 1000.0);
    }

    /* Phase 3: GPU marginal cost from slope (N=1 vs N=max). */
    if (num_dc >= 2) {
        double t1 = e2e_per_draw[0];
        double tN = e2e_per_draw[num_dc - 1];
        int N = draw_counts[num_dc - 1];
        double marginal_us = (tN * (double)N - t1) / (double)(N - 1) / 1000.0;
        record_result("GPU Fine: Marginal/draw", "us (slope)", "%.3f", marginal_us);
    }

    /* Phase 4: Command buffer commit overhead (fixed cost per flush).
     * Compare N draws + 1 query  vs  N * (1 draw + 1 query).
     * The difference is (N-1) extra query+flush overhead. */
    {
        int N = 32;
        uint64_t batch_total = 0, indiv_total = 0;

        for (int s = 0; s < samples; s++) {
            GLuint q;
            p_glGenQueries(1, &q);
            p_glBeginQuery(GL_TIME_ELAPSED, q);
            for (int i = 0; i < N; i++) {
                p_glDrawArrays(GL_TRIANGLES, 0, 3);
            }
            p_glEndQuery(GL_TIME_ELAPSED);
            GLuint64 result = 0;
            p_glGetQueryObjectui64v(q, GL_QUERY_RESULT, &result);
            p_glDeleteQueries(1, &q);
            batch_total += result;
        }
        for (int s = 0; s < samples; s++) {
            for (int i = 0; i < N; i++) {
                GLuint q;
                p_glGenQueries(1, &q);
                p_glBeginQuery(GL_TIME_ELAPSED, q);
                p_glDrawArrays(GL_TRIANGLES, 0, 3);
                p_glEndQuery(GL_TIME_ELAPSED);
                GLuint64 result = 0;
                p_glGetQueryObjectui64v(q, GL_QUERY_RESULT, &result);
                p_glDeleteQueries(1, &q);
                indiv_total += result;
            }
        }
        double commit_ns = indiv_total >= batch_total
            ? (double)(indiv_total - batch_total) / (double)samples / (double)(N - 1)
            : 0.0;
        record_result("GPU Fine: Per-query overhead", "us", "%.3f", commit_ns / 1000.0);
    }

    p_glDeleteBuffers(1, &vbo);
    p_glDeleteVertexArrays(1, &vao);
    p_glDeleteProgram(prog);
}

/* ======================================================================== */
/*  Benchmark 12: End-to-End                                                 */
/* ======================================================================== */

static void benchmark_end_to_end(void)
{
    GLuint prog_terrain = build_program(kVS2_src, kFS2_src);
    GLuint prog_entity  = build_program(kVS2_src, kFS2_src);
    GLuint prog_gui     = build_program(kVS_src,  kFS_src);

    GLint terrain_mvp   = p_glGetUniformLocation(prog_terrain, "uMVP");
    GLint terrain_color = p_glGetUniformLocation(prog_terrain, "uColor");
    GLint entity_mvp    = p_glGetUniformLocation(prog_entity,  "uMVP");
    GLint entity_color  = p_glGetUniformLocation(prog_entity,  "uColor");
    GLint gui_color     = p_glGetUniformLocation(prog_gui,     "uColor");

    const float cube_verts[] = {
        -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,
        -0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,
         0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f,  0.5f, -0.5f,
         0.5f, -0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f,
        -0.5f, -0.5f, -0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f,
        -0.5f, -0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f,
         0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,
         0.5f, -0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,
        -0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,
        -0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f,
        -0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f,
    };
    const int cube_vert_count = 36;

    const float quad_verts[] = {
        -0.5f, -0.5f, 0.0f,  0.5f, -0.5f, 0.0f,  0.5f,  0.5f, 0.0f,
        -0.5f, -0.5f, 0.0f,  0.5f,  0.5f, 0.0f, -0.5f,  0.5f, 0.0f,
    };
    const int quad_vert_count = 6;

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

    const int max_frames = g_frame_limit > 0 ? g_frame_limit : 3600;
    const uint64_t max_duration_ns = 60ULL * 1000ULL * 1000ULL * 1000ULL;

    for (int f = 0; f < g_warmup_frames; f++) {
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

        p_glEnable(GL_DEPTH_TEST);
        p_glUseProgram(prog_terrain);
        p_glBindVertexArray(vao);
        for (int i = 0; i < 100; i++) {
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

            p_glActiveTexture(GL_TEXTURE0);
            p_glBindTexture(GL_TEXTURE_2D, textures[i % num_tex]);

            if (i > 0 && (i % 10) == 0) {
                p_glDisable(GL_DEPTH_TEST);
                p_glEnable(GL_DEPTH_TEST);
            }
            if (i > 0 && (i % 20) == 0) {
                p_glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                p_glBlendFunc(GL_ONE, GL_ZERO);
            }

            p_glDrawArrays(GL_TRIANGLES, 0, cube_vert_count);
        }

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

        if ((frames % 30) == 0) {
            memset(tex_data, (uint8_t)(frames & 0xFF), tex_bytes);
            p_glActiveTexture(GL_TEXTURE0);
            p_glBindTexture(GL_TEXTURE_2D, textures[0]);
            p_glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tex_size, tex_size,
                              GL_RGBA, GL_UNSIGNED_BYTE, tex_data);
        }

        glfwSwapBuffers(g_window);

        uint64_t frame_end = now_ns();
        frame_times[frames] = (double)(frame_end - frame_start) / 1e6;
        frames++;

        if ((frame_end - loop_start) >= max_duration_ns) {
            break;
        }
    }

    record_frame_statistics("End-to-End", frame_times, frames, 140);

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
/*  Benchmark 13: Minecraft 1.21 CPU submission scene                        */
/* ======================================================================== */

/* This workload mirrors the resource shape used by Minecraft 1.21.11's
 * block/terrain, entity, and GUI paths: indexed draws, an aligned dynamic UBO
 * slice per draw, projection/fog UBOs, block atlas + lightmap sampling, entity
 * overlay sampling, and render-pass state changes. The scissor bounds fragment
 * work so the result remains primarily a CPU translation benchmark. */

#define MC_TERRAIN_DRAWS 896
#define MC_ENTITY_DRAWS  224
#define MC_GUI_DRAWS      96
#define MC_DRAWS_PER_FRAME (MC_TERRAIN_DRAWS + MC_ENTITY_DRAWS + MC_GUI_DRAWS)
#define MC_VAO_COUNT 12
#define MC_TEXTURE_COUNT 8
#define MC_DYNAMIC_BLOCK_SIZE 96

typedef struct {
    GLuint terrain_program;
    GLuint entity_program;
    GLuint gui_program;
    GLuint vaos[MC_VAO_COUNT];
    GLuint vertex_arena;
    GLsizeiptr vertex_slice_bytes;
    int has_vertex_binding_api;
    GLuint ebos[MC_VAO_COUNT];
    GLuint textures[MC_TEXTURE_COUNT];
    GLuint projection_ubo;
    GLuint fog_ubo;
    GLuint dynamic_ubo;
    GLint dynamic_stride;
    size_t dynamic_bytes;
    uint8_t *dynamic_data;
} MinecraftScene;

static const char *kMCVertexSource =
    "#version 330 core\n"
    "layout(location=0) in vec3 Position;\n"
    "layout(location=1) in vec4 Color;\n"
    "layout(location=2) in vec2 UV0;\n"
    "layout(std140) uniform Projection { mat4 ProjMat; };\n"
    "layout(std140) uniform DynamicTransforms {\n"
    "  mat4 ModelViewMat; vec4 ColorModulator; vec4 ModelOffset;\n"
    "};\n"
    "out vec4 vertexColor; out vec2 texCoord0; out float vertexDistance;\n"
    "void main() {\n"
    "  vec3 pos = Position + ModelOffset.xyz;\n"
    "  pos.x += float(gl_VertexID) * 0.0000001;\n"
    "  gl_Position = ProjMat * ModelViewMat * vec4(pos, 1.0);\n"
    "  vertexColor = Color * ColorModulator; texCoord0 = UV0;\n"
    "  vertexDistance = length(pos);\n"
    "}\n";

static const char *kMCTerrainFragmentSource =
    "#version 330 core\n"
    "uniform sampler2D Sampler0; uniform sampler2D Sampler2;\n"
    "layout(std140) uniform Fog { vec4 FogColor; vec4 FogParams; };\n"
    "in vec4 vertexColor; in vec2 texCoord0; in float vertexDistance;\n"
    "out vec4 fragColor;\n"
    "void main() {\n"
    "  vec4 color = texture(Sampler0, texCoord0) *\n"
    "               texture(Sampler2, vec2(0.5)) * vertexColor;\n"
    "  float fog = clamp((vertexDistance - FogParams.x) / FogParams.y, 0.0, 1.0);\n"
    "  fragColor = mix(color, FogColor, fog * FogColor.a);\n"
    "}\n";

static const char *kMCEntityFragmentSource =
    "#version 330 core\n"
    "uniform sampler2D Sampler0; uniform sampler2D Sampler1;\n"
    "uniform sampler2D Sampler2;\n"
    "layout(std140) uniform Fog { vec4 FogColor; vec4 FogParams; };\n"
    "in vec4 vertexColor; in vec2 texCoord0; in float vertexDistance;\n"
    "out vec4 fragColor;\n"
    "void main() {\n"
    "  vec4 base = texture(Sampler0, texCoord0);\n"
    "  vec4 overlay = texture(Sampler1, texCoord0);\n"
    "  vec4 color = mix(base, overlay, overlay.a * 0.25) *\n"
    "               texture(Sampler2, vec2(0.5)) * vertexColor;\n"
    "  float fog = clamp((vertexDistance - FogParams.x) / FogParams.y, 0.0, 1.0);\n"
    "  fragColor = mix(color, FogColor, fog * FogColor.a);\n"
    "}\n";

static const char *kMCGuiFragmentSource =
    "#version 330 core\n"
    "uniform sampler2D Sampler0;\n"
    "in vec4 vertexColor; in vec2 texCoord0; in float vertexDistance;\n"
    "out vec4 fragColor;\n"
    "void main() { fragColor = texture(Sampler0, texCoord0) * vertexColor; }\n";

static int bind_uniform_block(GLuint program, const char *name, GLuint binding)
{
    GLuint index = p_glGetUniformBlockIndex(program, name);
    if (index == GL_INVALID_INDEX) {
        fprintf(stderr, "  [warning] uniform block '%s' is inactive\n", name);
        return 0;
    }
    p_glUniformBlockBinding(program, index, binding);
    return 1;
}

static void set_sampler_unit(GLuint program, const char *name, GLint unit)
{
    GLint location = p_glGetUniformLocation(program, name);
    if (location >= 0) p_glUniform1i(location, unit);
}

static void destroy_minecraft_scene(MinecraftScene *scene)
{
    if (!scene) return;
    if (scene->dynamic_ubo) p_glDeleteBuffers(1, &scene->dynamic_ubo);
    if (scene->fog_ubo) p_glDeleteBuffers(1, &scene->fog_ubo);
    if (scene->projection_ubo) p_glDeleteBuffers(1, &scene->projection_ubo);
    if (scene->textures[0]) p_glDeleteTextures(MC_TEXTURE_COUNT, scene->textures);
    if (scene->ebos[0]) p_glDeleteBuffers(MC_VAO_COUNT, scene->ebos);
    if (scene->vertex_arena) p_glDeleteBuffers(1, &scene->vertex_arena);
    if (scene->vaos[0]) p_glDeleteVertexArrays(MC_VAO_COUNT, scene->vaos);
    if (scene->gui_program) p_glDeleteProgram(scene->gui_program);
    if (scene->entity_program) p_glDeleteProgram(scene->entity_program);
    if (scene->terrain_program) p_glDeleteProgram(scene->terrain_program);
    free(scene->dynamic_data);
    memset(scene, 0, sizeof(*scene));
}

static int init_minecraft_scene(MinecraftScene *scene)
{
    memset(scene, 0, sizeof(*scene));
    scene->terrain_program = build_program(kMCVertexSource, kMCTerrainFragmentSource);
    scene->entity_program = build_program(kMCVertexSource, kMCEntityFragmentSource);
    scene->gui_program = build_program(kMCVertexSource, kMCGuiFragmentSource);
    if (!scene->terrain_program || !scene->entity_program || !scene->gui_program)
        return 0;

    GLuint programs[] = {
        scene->terrain_program, scene->entity_program, scene->gui_program
    };
    for (int i = 0; i < 3; i++) {
        bind_uniform_block(programs[i], "Projection", 0);
        bind_uniform_block(programs[i], "DynamicTransforms", 1);
        if (i < 2) bind_uniform_block(programs[i], "Fog", 2);
    }

    p_glUseProgram(scene->terrain_program);
    set_sampler_unit(scene->terrain_program, "Sampler0", 0);
    set_sampler_unit(scene->terrain_program, "Sampler2", 2);
    p_glUseProgram(scene->entity_program);
    set_sampler_unit(scene->entity_program, "Sampler0", 0);
    set_sampler_unit(scene->entity_program, "Sampler1", 1);
    set_sampler_unit(scene->entity_program, "Sampler2", 2);
    p_glUseProgram(scene->gui_program);
    set_sampler_unit(scene->gui_program, "Sampler0", 0);

    static const float base_vertices[] = {
        -0.04f, -0.04f, 0.0f, 1, 1, 1, 1, 0, 0,
         0.04f, -0.04f, 0.0f, 1, 1, 1, 1, 1, 0,
         0.04f,  0.04f, 0.0f, 1, 1, 1, 1, 1, 1,
        -0.04f,  0.04f, 0.0f, 1, 1, 1, 1, 0, 1,
    };
    static const GLuint indices[] = {0, 1, 2, 0, 2, 3};
    p_glGenBuffers(MC_VAO_COUNT, scene->ebos);
    for (int i = 0; i < MC_VAO_COUNT; i++) {
        p_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scene->ebos[i]);
        p_glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
                       GL_STATIC_DRAW);
    }

    float vertex_arena[MC_VAO_COUNT]
                      [sizeof(base_vertices) / sizeof(base_vertices[0])];
    for (int i = 0; i < MC_VAO_COUNT; i++) {
        memcpy(vertex_arena[i], base_vertices, sizeof(base_vertices));
        for (int v = 0; v < 4; v++) {
            vertex_arena[i][v * 9 + 2] = (float)i * 0.0001f;
        }
    }
    scene->vertex_slice_bytes = (GLsizeiptr)sizeof(base_vertices);
    p_glGenBuffers(1, &scene->vertex_arena);
    p_glBindBuffer(GL_ARRAY_BUFFER, scene->vertex_arena);
    p_glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_arena), vertex_arena,
                   GL_STATIC_DRAW);

    scene->has_vertex_binding_api =
        p_glBindVertexBuffer && p_glVertexAttribFormat && p_glVertexAttribBinding;
    p_glGenVertexArrays(MC_VAO_COUNT, scene->vaos);
    for (int i = 0; i < MC_VAO_COUNT; i++) {
        p_glBindVertexArray(scene->vaos[i]);
        p_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scene->ebos[i]);
        p_glEnableVertexAttribArray(0);
        p_glEnableVertexAttribArray(1);
        p_glEnableVertexAttribArray(2);
        if (scene->has_vertex_binding_api) {
            p_glVertexAttribFormat(0, 3, GL_FLOAT, GL_FALSE, 0);
            p_glVertexAttribFormat(1, 4, GL_FLOAT, GL_FALSE, 3 * sizeof(float));
            p_glVertexAttribFormat(2, 2, GL_FLOAT, GL_FALSE, 7 * sizeof(float));
            p_glVertexAttribBinding(0, 0);
            p_glVertexAttribBinding(1, 0);
            p_glVertexAttribBinding(2, 0);
            p_glBindVertexBuffer(0, scene->vertex_arena,
                                 (GLintptr)((size_t)i * scene->vertex_slice_bytes),
                                 9 * sizeof(float));
        } else {
            uintptr_t base = (uintptr_t)((size_t)i * scene->vertex_slice_bytes);
            p_glBindBuffer(GL_ARRAY_BUFFER, scene->vertex_arena);
            p_glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float),
                                    (void *)base);
            p_glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 9 * sizeof(float),
                                    (void *)(base + 3 * sizeof(float)));
            p_glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 9 * sizeof(float),
                                    (void *)(base + 7 * sizeof(float)));
        }
    }

    uint8_t pixels[16 * 16 * 4];
    p_glGenTextures(MC_TEXTURE_COUNT, scene->textures);
    for (int i = 0; i < MC_TEXTURE_COUNT; i++) {
        for (size_t p = 0; p < sizeof(pixels); p += 4) {
            pixels[p + 0] = (uint8_t)(48 + i * 23);
            pixels[p + 1] = (uint8_t)(180 - i * 13);
            pixels[p + 2] = (uint8_t)(96 + i * 11);
            pixels[p + 3] = (uint8_t)(i == 6 ? 96 : 255);
        }
        p_glActiveTexture(GL_TEXTURE0);
        p_glBindTexture(GL_TEXTURE_2D, scene->textures[i]);
        p_glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 16, 16, 0,
                       GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    const float identity[16] = {
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1
    };
    const float fog[8] = {0.55f, 0.68f, 0.80f, 0.35f, 0.2f, 5.0f, 0, 0};
    p_glGenBuffers(1, &scene->projection_ubo);
    p_glBindBuffer(GL_UNIFORM_BUFFER, scene->projection_ubo);
    p_glBufferData(GL_UNIFORM_BUFFER, sizeof(identity), identity, GL_STATIC_DRAW);
    p_glGenBuffers(1, &scene->fog_ubo);
    p_glBindBuffer(GL_UNIFORM_BUFFER, scene->fog_ubo);
    p_glBufferData(GL_UNIFORM_BUFFER, sizeof(fog), fog, GL_STATIC_DRAW);

    GLint alignment = 1;
    p_glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &alignment);
    if (alignment < 1) alignment = 1;
    scene->dynamic_stride = (MC_DYNAMIC_BLOCK_SIZE + alignment - 1) / alignment * alignment;
    scene->dynamic_bytes = (size_t)scene->dynamic_stride * MC_DRAWS_PER_FRAME;
    scene->dynamic_data = (uint8_t *)calloc(1, scene->dynamic_bytes);
    if (!scene->dynamic_data) return 0;
    for (int i = 0; i < MC_DRAWS_PER_FRAME; i++) {
        float *block = (float *)(scene->dynamic_data + (size_t)i * scene->dynamic_stride);
        memcpy(block, identity, sizeof(identity));
        block[12] = (float)((i % 32) - 16) * 0.015f;
        block[13] = (float)(((i / 32) % 24) - 12) * 0.015f;
        block[16] = 0.75f + (float)(i % 5) * 0.04f;
        block[17] = 0.82f;
        block[18] = 0.70f;
        block[19] = 1.0f;
    }
    p_glGenBuffers(1, &scene->dynamic_ubo);
    p_glBindBuffer(GL_UNIFORM_BUFFER, scene->dynamic_ubo);
    p_glBufferData(GL_UNIFORM_BUFFER, scene->dynamic_bytes,
                   scene->dynamic_data, GL_STREAM_DRAW);

    p_glBindBufferRange(GL_UNIFORM_BUFFER, 0, scene->projection_ubo, 0,
                        sizeof(identity));
    p_glBindBufferRange(GL_UNIFORM_BUFFER, 2, scene->fog_ubo, 0, sizeof(fog));
    return p_glGetError() == GL_NO_ERROR;
}

static void draw_minecraft_cpu_frame(MinecraftScene *scene, int frame)
{
    p_glBindBuffer(GL_UNIFORM_BUFFER, scene->dynamic_ubo);
    scene->dynamic_data[19 * sizeof(float)] = (uint8_t)(frame & 0xff);
    p_glBufferSubData(GL_UNIFORM_BUFFER, 0, scene->dynamic_bytes,
                      scene->dynamic_data);

    p_glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    p_glEnable(GL_SCISSOR_TEST);
    p_glScissor(0, 0, 8, 8);
    p_glEnable(GL_DEPTH_TEST);
    p_glDisable(GL_BLEND);

    if (scene->has_vertex_binding_api) p_glBindVertexArray(scene->vaos[0]);

    p_glUseProgram(scene->terrain_program);
    p_glActiveTexture(GL_TEXTURE0);
    p_glBindTexture(GL_TEXTURE_2D, scene->textures[0]);
    p_glActiveTexture(GL_TEXTURE2);
    p_glBindTexture(GL_TEXTURE_2D, scene->textures[7]);
    for (int i = 0; i < MC_TERRAIN_DRAWS; i++) {
        int geometry = i % MC_VAO_COUNT;
        if (scene->has_vertex_binding_api) {
            p_glBindVertexBuffer(0, scene->vertex_arena,
                                 (GLintptr)((size_t)geometry * scene->vertex_slice_bytes),
                                 9 * sizeof(float));
            p_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scene->ebos[geometry]);
        } else {
            p_glBindVertexArray(scene->vaos[geometry]);
        }
        p_glBindBufferRange(GL_UNIFORM_BUFFER, 1, scene->dynamic_ubo,
                            (GLintptr)((size_t)i * scene->dynamic_stride),
                            MC_DYNAMIC_BLOCK_SIZE);
        p_glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void *)0);
    }

    p_glUseProgram(scene->entity_program);
    p_glActiveTexture(GL_TEXTURE1);
    p_glBindTexture(GL_TEXTURE_2D, scene->textures[6]);
    for (int i = 0; i < MC_ENTITY_DRAWS; i++) {
        int draw_index = MC_TERRAIN_DRAWS + i;
        p_glActiveTexture(GL_TEXTURE0);
        p_glBindTexture(GL_TEXTURE_2D, scene->textures[1 + (i % 5)]);
        int geometry = (i * 5) % MC_VAO_COUNT;
        if (scene->has_vertex_binding_api) {
            p_glBindVertexBuffer(0, scene->vertex_arena,
                                 (GLintptr)((size_t)geometry * scene->vertex_slice_bytes),
                                 9 * sizeof(float));
            p_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scene->ebos[geometry]);
        } else {
            p_glBindVertexArray(scene->vaos[geometry]);
        }
        p_glBindBufferRange(GL_UNIFORM_BUFFER, 1, scene->dynamic_ubo,
                            (GLintptr)((size_t)draw_index * scene->dynamic_stride),
                            MC_DYNAMIC_BLOCK_SIZE);
        p_glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void *)0);
    }

    p_glDisable(GL_DEPTH_TEST);
    p_glEnable(GL_BLEND);
    p_glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    p_glUseProgram(scene->gui_program);
    for (int i = 0; i < MC_GUI_DRAWS; i++) {
        int draw_index = MC_TERRAIN_DRAWS + MC_ENTITY_DRAWS + i;
        if ((i % 8) == 0) {
            p_glActiveTexture(GL_TEXTURE0);
            p_glBindTexture(GL_TEXTURE_2D, scene->textures[(i / 8) % 4]);
        }
        if (scene->has_vertex_binding_api) {
            p_glBindVertexBuffer(0, scene->vertex_arena, 0,
                                 9 * sizeof(float));
            p_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
                           scene->ebos[i % MC_VAO_COUNT]);
        } else {
            p_glBindVertexArray(scene->vaos[0]);
        }
        p_glBindBufferRange(GL_UNIFORM_BUFFER, 1, scene->dynamic_ubo,
                            (GLintptr)((size_t)draw_index * scene->dynamic_stride),
                            MC_DYNAMIC_BLOCK_SIZE);
        p_glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void *)0);
    }
    p_glDisable(GL_BLEND);
    p_glDisable(GL_SCISSOR_TEST);
}

static int benchmark_minecraft_cpu(void)
{
    MinecraftScene scene;
    if (!init_minecraft_scene(&scene)) {
        fprintf(stderr, "  [warning] failed to initialize Minecraft CPU scene\n");
        destroy_minecraft_scene(&scene);
        return 0;
    }

    p_glViewport(0, 0, 640, 480);
    p_glClearColor(0.05f, 0.07f, 0.09f, 1.0f);
    for (int i = 0; i < g_warmup_frames; i++) {
        draw_minecraft_cpu_frame(&scene, i);
        glfwSwapBuffers(g_window);
    }

    int frames = g_frame_limit > 0 ? g_frame_limit : 120;
    double *frame_times = (double *)malloc((size_t)frames * sizeof(double));
    double *submit_times = (double *)malloc((size_t)frames * sizeof(double));
    double *swap_times = (double *)malloc((size_t)frames * sizeof(double));
    if (!frame_times || !submit_times || !swap_times) {
        fprintf(stderr, "  [warning] failed to allocate Minecraft CPU timing buffers\n");
        free(swap_times);
        free(submit_times);
        free(frame_times);
        destroy_minecraft_scene(&scene);
        return 0;
    }
    for (int frame = 0; frame < frames; frame++) {
        uint64_t start = now_ns();
        draw_minecraft_cpu_frame(&scene, frame);
        uint64_t submit_end = now_ns();
        glfwSwapBuffers(g_window);
        uint64_t frame_end = now_ns();
        submit_times[frame] = (double)(submit_end - start) / 1e6;
        swap_times[frame] = (double)(frame_end - submit_end) / 1e6;
        frame_times[frame] = (double)(frame_end - start) / 1e6;
    }

    record_duration_statistics("Minecraft CPU 1.21", "CPU Submit",
                               submit_times, frames);
    record_duration_statistics("Minecraft CPU 1.21", "Swap",
                               swap_times, frames);
    record_frame_statistics("Minecraft CPU 1.21", frame_times, frames,
                            MC_DRAWS_PER_FRAME);
    free(swap_times);
    free(submit_times);
    free(frame_times);
    destroy_minecraft_scene(&scene);
    return 1;
}

/* ======================================================================== */
/*  Main                                                                     */
/* ======================================================================== */

static void print_usage(const char *prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("\n");
    printf("Runs the benchmark suite and prints a results table at the\n");
    printf("end.  The program creates a hidden 640x480 GLFW window, loads\n");
    printf("GL entry points through glfwGetProcAddress, and measures\n");
    printf("translation overhead and GPU execution time.\n");
    printf("\n");
    printf("Backend: %s\n", BENCHMARK_BACKEND_NAME);
    printf("\n");
    printf("Categories:\n");
    printf("  1.  Empty Draw         7.  Pipeline Switch\n");
    printf("  2.  Triangle Draw      8.  Uniform Update\n");
    printf("  3.  Batch Draw         9.  GPU Time (Timer Query)\n");
    printf("  4.  Texture Upload    10.  End-to-End (Minecraft-like)\n");
    printf("  5.  Buffer Upload     11.  API Dispatch (no-op overhead)\n");
    printf("  6.  State Changes     12.  Fine-grained GPU Time (Timer Query)\n");
    printf(" 13.  Minecraft CPU 1.21 submission scene\n");
    printf("\n");
    printf("Options:\n");
    printf("  --test NAME      Run one test (use --list to show names)\n");
    printf("  --frames N       Override measured frame count for frame tests\n");
    printf("  --warmup N       Set warm-up frame count (default: 10)\n");
    printf("  --json PATH      Write metadata and numeric results as JSON\n");
    printf("  --list           List selectable test names\n");
    printf("  --help, -h       Show this help\n");
}

static const char *const kTestNames[] = {
    "empty", "triangle", "batch", "texture-upload", "buffer-upload",
    "state", "pipeline", "uniform", "gpu", "dispatch", "gpu-fine",
    "minecraft-cpu", "end-to-end"
};

static void print_test_names(void)
{
    for (size_t i = 0; i < sizeof(kTestNames) / sizeof(kTestNames[0]); i++)
        printf("%s\n", kTestNames[i]);
}

static int valid_test_name(const char *name)
{
    for (size_t i = 0; i < sizeof(kTestNames) / sizeof(kTestNames[0]); i++) {
        if (strcmp(name, kTestNames[i]) == 0) return 1;
    }
    return 0;
}

static int parse_nonnegative_int(const char *option, const char *value, int *out)
{
    char *end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno != 0 || !end || *end != '\0' || parsed < 0 || parsed > 1000000) {
        fprintf(stderr, "Invalid value for %s: %s\n", option, value);
        return 0;
    }
    *out = (int)parsed;
    return 1;
}

/* ======================================================================== */
/*  Context lifecycle + benchmark runner (extracted for A/B mode)            */
/* ======================================================================== */

static int create_context(void)
{
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    g_window = glfwCreateWindow(640, 480, "mgl_benchmark", NULL, NULL);
    if (!g_window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        return 0;
    }

    glfwMakeContextCurrent(g_window);
    glfwSwapInterval(0);

    if (!load_gl()) {
        fprintf(stderr, "Failed to load GL functions\n");
        return 0;
    }

    const GLubyte *renderer = p_glGetString(GL_RENDERER);
    const GLubyte *version = p_glGetString(GL_VERSION);
    const GLubyte *vendor = p_glGetString(GL_VENDOR);
    g_renderer_name = renderer ? (const char *)renderer : "(null)";
    g_version_name = version ? (const char *)version : "(null)";
    g_vendor_name = vendor ? (const char *)vendor : "(null)";

    p_glViewport(0, 0, 640, 480);
    check_gl_error("initial setup");
    return 1;
}

static void destroy_context(void)
{
    if (g_window) {
        glfwDestroyWindow(g_window);
        g_window = NULL;
    }
}

static int run_all_benchmarks(void)
{
    int succeeded = 1;

    if (should_run("empty")) benchmark_empty_draw();
    if (should_run("triangle")) benchmark_triangle_draw();
    if (should_run("batch")) benchmark_batch_draw();
    if (should_run("texture-upload")) benchmark_texture_upload();
    if (should_run("buffer-upload")) benchmark_buffer_upload();
    if (should_run("state")) benchmark_state_changes();
    if (should_run("pipeline")) benchmark_pipeline_switch();
    if (should_run("uniform")) benchmark_uniform_update();
    if (should_run("gpu")) benchmark_gpu_time();
    if (should_run("dispatch")) benchmark_api_dispatch();
    if (should_run("gpu-fine")) benchmark_gpu_time_fine();
    if (should_run("minecraft-cpu") && !benchmark_minecraft_cpu()) {
        succeeded = 0;
    }
    if (should_run("end-to-end")) benchmark_end_to_end();

    return succeeded;
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--list") == 0) {
            print_test_names();
            return 0;
        } else if (strcmp(argv[i], "--test") == 0 && i + 1 < argc) {
            g_selected_test = argv[++i];
            if (!valid_test_name(g_selected_test)) {
                fprintf(stderr, "Unknown test: %s\n", g_selected_test);
                print_test_names();
                return 1;
            }
        } else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            if (!parse_nonnegative_int("--frames", argv[++i], &g_frame_limit) ||
                g_frame_limit == 0) return 1;
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            if (!parse_nonnegative_int("--warmup", argv[++i], &g_warmup_frames))
                return 1;
        } else if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            g_json_path = argv[++i];
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    init_timing();

    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }

    int benchmarks_succeeded = 1;

    if (!create_context()) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return 1;
    }

    printf("Backend:   %s\n", BENCHMARK_BACKEND_NAME);
    printf("Renderer:  %s\n", g_renderer_name);
    printf("Version:   %s\n", g_version_name);
    printf("Vendor:    %s\n", g_vendor_name);
    printf("\n");

    printf("Running benchmarks...\n");
    if (!run_all_benchmarks()) benchmarks_succeeded = 0;

    print_all_results();
    destroy_context();
    glfwTerminate();

    if (g_json_path && !write_json_results(g_json_path)) {
        return 1;
    }

    return benchmarks_succeeded ? 0 : 1;
}
